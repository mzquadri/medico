"""
PHASE 4: MASKED FOCAL LOSS FINE-TUNING
========================================
Fine-tune Phase 3 checkpoint with uncertainty masking:

1. MaskedSmoothedFocalLoss: Ignore uncertain (-1) labels in CheXpert
2. Mask-aware mixup: Handle mixed masks correctly
3. CheXpert NaN→0, -1→masked (not false negative)
4. NIH & Pneumonia: Always return masks (all 1s)
5. Fine-tune from Epoch 16 checkpoint
6. Low LR (5e-6), low dropout (0.25), no mixup
7. CPU-friendly: No gradient checkpointing

Target: Fix Cardiomegaly performance by handling uncertain labels correctly
"""

import os
import sys
import warnings
from datetime import datetime
import random
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.checkpoint import checkpoint_sequential
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from torch.cuda.amp import autocast, GradScaler
    HAS_AMP = True
except ImportError:
    HAS_AMP = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ============================================================================
# VERSION-SAFE DATALOADER HELPER
# ============================================================================
def create_dataloader(dataset, batch_size, num_workers=0, shuffle=False, sampler=None, 
                     pin_memory=False, prefetch_factor=2, persistent_workers=False):
    """
    Create DataLoader with version-safe parameters.
    Automatically handles NUM_WORKERS=0 case where prefetch_factor/persistent_workers must be omitted.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data (ignored if sampler is provided)
        sampler: Optional sampler
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch (only used if num_workers > 0)
        persistent_workers: Keep workers alive between epochs (only used if num_workers > 0)
    
    Returns:
        DataLoader: Version-safe DataLoader instance
    """
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle if sampler is None else False,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
    
    if sampler is not None:
        kwargs['sampler'] = sampler
    
    # Only add these parameters if num_workers > 0 (avoids TypeError on some PyTorch versions)
    if num_workers > 0:
        kwargs['prefetch_factor'] = prefetch_factor
        kwargs['persistent_workers'] = persistent_workers
    
    return DataLoader(dataset, **kwargs)

# Device Setup - Intel Arc GPU Support
# Priority: Intel XPU (native) > DirectML > CUDA > CPU
try:
    # Try Intel Extension for PyTorch (IPEX) - best for Intel Arc
    import intel_extension_for_pytorch as ipex
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        DEVICE = torch.device('xpu')
        IS_DIRECTML = False
        print(f"Using Intel XPU: {DEVICE}")
        print(f"Intel Arc Device: {torch.xpu.get_device_name(0)}")
        print("Note: AMP disabled for Intel XPU")
    else:
        raise ImportError("Intel XPU not available")
except ImportError:
    try:
        # Fallback to DirectML (Windows only)
        import torch_directml
        DEVICE = torch_directml.device()
        IS_DIRECTML = True
        print(f"Using DirectML: {DEVICE} (Intel Arc)")
        print("Note: AMP disabled for DirectML compatibility")
        print("WARNING: DirectML may have performance issues with convolutions")
    except ImportError:
        # Final fallback: CUDA or CPU
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using Device: {DEVICE}")
        IS_DIRECTML = False

# AMP only enabled on CUDA, not DirectML
AMP_ENABLED = (HAS_AMP and str(DEVICE) == "cuda" and torch.cuda.is_available() and not IS_DIRECTML)

# ============================================================================
# PHASE 4 FINE-TUNING CONFIGURATION
# ============================================================================
MODEL_VERSION = "3.1.0-finetune-masked"
MODEL_NAME = "ChestXray-DenseNet121-Phase4-MaskedFineTune-14diseases"

# ALL 14 NIH DISEASES (ordered alphabetically for consistency)
SELECTED_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",          # NIH name (will map to Pleural Effusion in display)
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax"
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(SELECTED_LABELS)}  # Safe label indexing

# FINE-TUNING HYPERPARAMETERS (optimized for Phase 4)
IMG_SIZE = 320
BATCH_SIZE = 8             # Keep same for consistency
ACCUMULATION_STEPS = 16      # Effective batch = 128
LEARNING_RATE = 5e-6         # REDUCED for fine-tuning (was 3e-5)
FINETUNE_EPOCHS = 15         # How many NEW epochs to run after resume
EARLY_STOPPING_PATIENCE = 10 # Reduced patience
DROPOUT = 0.25               # REDUCED from 0.5 (less regularization)
MIXUP_ALPHA = 0.0            # DISABLED for fine-tuning (was 0.15)

# DATA LOADING OPTIMIZATION
# Windows spawn safety: limit workers to avoid hangs
if os.name == "nt":  # Windows
    NUM_WORKERS = 0 if IS_DIRECTML else 2  # Conservative for Windows stability
    PERSISTENT_WORKERS = False  # Safer with spawn on Windows
    PREFETCH_FACTOR = 2
else:  # Linux/Unix
    NUM_WORKERS = 4 if IS_DIRECTML else 8  # More workers on CUDA for better throughput
    PREFETCH_FACTOR = 2 if IS_DIRECTML else 4  # Higher prefetch on CUDA
    PERSISTENT_WORKERS = not IS_DIRECTML  # Enable on CUDA for faster dataloading

PIN_MEMORY = (str(DEVICE) == "cuda")  # Only enable on CUDA, not DirectML or CPU

# TRAINING EFFICIENCY
VALIDATE_EVERY_N_EPOCHS = 2  # Validate less frequently
GRADIENT_CLIP_NORM = 1.0     # Gradient clipping
USE_TTA = False              # Test-Time Augmentation (horizontal flip, ~2x val time, +AUC)

# FULL DATASET TRAINING (NO SAMPLING)
CHEXPERT_SAMPLE_SIZE = None  # Use all 223K images
NIH_SAMPLE_SIZE = None       # Use all 112K images
USE_FULL_PNEUMONIA = True    # Use all 5.2K images

# Gradient Checkpointing (Memory Efficiency)
USE_GRADIENT_CHECKPOINTING = False  # Disabled for Phase 4 (CPU-friendly fine-tuning)

# Focal Loss
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Paths
CHEXPERT_ROOT = r"C:\Users\MohdZaminQuadri\Downloads\Medico-Xray\datasets\chexpert"
CHEXPERT_TRAIN_CSV = os.path.join(CHEXPERT_ROOT, "train.csv")  # Will be split 80/10/10
CHEXPERT_VALID_CSV = os.path.join(CHEXPERT_ROOT, "valid.csv")  # Original small validation (unused)

NIH_IMAGE_DIR = r"C:\Users\MohdZaminQuadri\Downloads\Medico-Xray\datasets\nih\images\images"
# Fixed: CSV file is inside Data_Entry_2017.csv directory (directory name, not file!)
NIH_CSV_PATH = r"C:\Users\MohdZaminQuadri\Downloads\Medico-Xray\datasets\nih\Data_Entry_2017.csv\Data_Entry_2017.csv"

PNEUMONIA_ROOT = r"C:\Users\MohdZaminQuadri\Downloads\Medico-Xray\datasets\pneumonia"

PHASE3_DIR = "checkpoints_phase3_fulldata"
CHECKPOINT_DIR = "checkpoints_phase4_masked"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_FILE = os.path.join(CHECKPOINT_DIR, 'training_log.txt')
DETAILED_LOG_FILE = os.path.join(CHECKPOINT_DIR, 'training_log_detailed.json')

# RESUME FROM BEST CHECKPOINT
RESUME_FROM_BEST = True  # PHASE 4: Fine-tune from Phase 3 Epoch 16 checkpoint
BEST_CHECKPOINT = os.path.join(PHASE3_DIR, "best_model_phase3_fulldata.pt")
BEST_MODEL_OUT = os.path.join(CHECKPOINT_DIR, "best_model_phase4_masked.pt")

LOG_DIR = "logs_phase3"
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(DEVICE) == "cuda":
        torch.cuda.manual_seed_all(seed)
        # Performance optimization: allow fast kernels and better matmul (CUDA only)
        torch.backends.cudnn.benchmark = True  # Enable for max speed on CUDA
        torch.backends.cudnn.deterministic = False  # Disable for max speed
        # TF32 for matmul (Ampere+ GPUs, ~20% speedup with minimal accuracy impact)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Better matmul performance (Ampere+ GPUs)
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('medium')
    elif IS_DIRECTML:
        print("Using DirectML - cudnn.benchmark disabled for compatibility")
        print("WARNING: DirectML with NUM_WORKERS=0 will bottleneck on CPU transforms (CLAHE, PIL IO)")
        print("Consider: (1) Disable CLAHE during training, or (2) Precompute CLAHE+resize offline")

set_seed(42)
print("Random seeds set for reproducibility\n")
if str(DEVICE) == "cuda":
    if torch.backends.cudnn.benchmark:
        print("CUDA optimizations enabled (cudnn.benchmark=True, matmul=medium)\n")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def validate_paths():
    """Validate all required dataset paths exist"""
    critical_paths = [
        CHEXPERT_TRAIN_CSV,
        # CHEXPERT_VALID_CSV removed - not used in Phase 4 (we create our own splits)
        NIH_CSV_PATH,
        NIH_IMAGE_DIR,
        PNEUMONIA_ROOT
    ]
    
    for path in critical_paths:
        if not os.path.exists(path):
            print(f"ERROR: Required path not found: {path}")
            sys.exit(1)
    print("All dataset paths validated\n")

def print_memory_info():
    """Print current memory usage"""
    if str(DEVICE) == "cuda":
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
    elif IS_DIRECTML:
        print("DirectML device active")
    else:
        if HAS_PSUTIL:
            memory_gb = psutil.virtual_memory().used / 1e9
            print(f"CPU Memory Used: {memory_gb:.1f}GB")
        else:
            print("CPU Memory monitoring unavailable (install psutil)")

def print_system_info():
    """Print comprehensive system information"""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"DirectML: {IS_DIRECTML}")
    print(f"AMP Enabled: {AMP_ENABLED}")
    
    if str(DEVICE) == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f}GB")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version.split()[0]}")
    print("="*70 + "\n")

def save_training_config():
    """Save complete training configuration to JSON"""
    config = {
        'MODEL_VERSION': MODEL_VERSION,
        'MODEL_NAME': MODEL_NAME,
        'SELECTED_LABELS': SELECTED_LABELS,
        'IMG_SIZE': IMG_SIZE,
        'BATCH_SIZE': BATCH_SIZE,
        'ACCUMULATION_STEPS': ACCUMULATION_STEPS,
        'LEARNING_RATE': LEARNING_RATE,
        'FINETUNE_EPOCHS': FINETUNE_EPOCHS,
        'DROPOUT': DROPOUT,
        'MIXUP_ALPHA': MIXUP_ALPHA,
        'NUM_WORKERS': NUM_WORKERS,
        'PREFETCH_FACTOR': PREFETCH_FACTOR,
        'FOCAL_ALPHA': FOCAL_ALPHA,
        'FOCAL_GAMMA': FOCAL_GAMMA,
        'USE_GRADIENT_CHECKPOINTING': USE_GRADIENT_CHECKPOINTING,
        'DEVICE': str(DEVICE),
        'AMP_ENABLED': AMP_ENABLED,
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = os.path.join(CHECKPOINT_DIR, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}\n")

def print_dataset_stats(dataset_name, dataset, selected_labels):
    """Print dataset statistics for analysis"""
    print(f"\n{dataset_name} Dataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    
    # Try precomputed labels first (NIH, CheXpert)
    if hasattr(dataset, 'labels'):
        labels_matrix = dataset.labels
        for i, label in enumerate(selected_labels):
            positive = int(labels_matrix[:, i].sum())
            total = len(dataset)
            print(f"  {label:20s}: {positive:6d}/{total} ({positive/total*100:.1f}%)")
    # Fallback to df columns (old CheXpert path)
    elif hasattr(dataset, 'df'):
        df = dataset.df
        for label in selected_labels:
            if label in df.columns:
                positive = int((df[label] == 1).sum())
                total = len(df)
                print(f"  {label:20s}: {positive:6d}/{total} ({positive/total*100:.1f}%)")
    # Pneumonia dataset
    elif hasattr(dataset, 'labels_list'):
        positive = int(sum(dataset.labels_list))
        total = len(dataset.labels_list)
        print(f"  Pneumonia cases: {positive:6d}/{total} ({positive/total*100:.1f}%)")

def quick_image_path_sanity(dataset, name, n=2000):
    """
    Quick sanity check to verify image paths exist.
    Prevents silent training on blank images due to path mismatches.
    
    Args:
        dataset: Dataset to check (CheXpert, NIH, or Pneumonia)
        name: Dataset name for logging
        n: Number of samples to check (default 2000)
    """
    missing = 0
    
    # CheXpert or NIH datasets
    if hasattr(dataset, 'df'):
        total = min(n, len(dataset.df))
        sample_indices = np.random.choice(len(dataset.df), size=total, replace=False)
        
        for i in sample_indices:
            row = dataset.df.iloc[i]
            
            # CheXpert path handling - match CheXpertDataset logic (try both strategies)
            if hasattr(dataset, 'root_dir'):
                rel_path = row['Path']
                rel_path_normalized = rel_path.replace('/', os.sep)
                img_path = os.path.join(dataset.root_dir, rel_path_normalized)
                
                # If not found and has prefix, try stripping it
                if not os.path.exists(img_path) and rel_path.startswith('CheXpert-v1.0-small/'):
                    rel_path_stripped = rel_path.replace('CheXpert-v1.0-small/', '', 1).replace('/', os.sep)
                    img_path = os.path.join(dataset.root_dir, rel_path_stripped)
            # NIH path handling
            elif hasattr(dataset, 'image_dir'):
                img_path = os.path.join(dataset.image_dir, row['Image Index'])
            else:
                continue
            
            if not os.path.exists(img_path):
                missing += 1
    
    # Pneumonia dataset
    elif hasattr(dataset, 'image_paths'):
        total = min(n, len(dataset.image_paths))
        sample_indices = np.random.choice(len(dataset.image_paths), size=total, replace=False)
        
        for i in sample_indices:
            img_path = dataset.image_paths[i]
            if not os.path.exists(img_path):
                missing += 1
    else:
        print(f"{name}: No df/image_paths found, skipping path sanity check")
        return
    
    miss_pct = 100 * missing / total
    print(f"{name} path sanity: missing {missing}/{total} ({miss_pct:.2f}%)")
    
    if miss_pct > 1.0:
        print(f"  ERROR: {miss_pct:.2f}% images missing!")
        print(f"  Check root_dir/prefix stripping/path separators")
        raise RuntimeError(
            f"{name}: {miss_pct:.2f}% images missing. Training would learn from blank fallback images. "
            f"Fix dataset paths before starting training."
        )
    elif missing > 0:
        print(f"  Minor: {missing} images missing ({miss_pct:.2f}%), will use blank fallbacks")
    else:
        print(f"  All sampled paths verified")

def validate_training_setup(model, train_loader, criterion, optimizer):
    """Test training setup with one batch (no weight update)"""
    print("\nValidating training setup...")
    model.train()
    
    try:
        batch = next(iter(train_loader))
        
        # Handle 3-tuple (Phase 4) or 2-tuple (legacy)
        if len(batch) == 3:
            images, labels, masks = batch
            masks = masks.to(DEVICE)
        else:
            images, labels = batch
            masks = None
        
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels, masks)  # Pass masks
        
        optimizer.zero_grad()
        loss.backward()
        # CRITICAL: Don't step optimizer (would change weights before training starts)
        optimizer.zero_grad(set_to_none=True)  # Clear gradients without updating
        
        print("Training setup validation passed\n")
        return True
        
    except Exception as e:
        print(f"ERROR: Training setup validation failed: {e}")
        return False

def estimate_dataset_normalization(dataset, sample_size=1000):
    """Calculate mean and std for medical dataset normalization"""
    print(f"Calculating dataset statistics (sampling {sample_size} images)...")
    
    # Create temporary loader without normalization
    temp_transform = transforms.Compose([
        transforms.Resize(340),
        transforms.CenterCrop(IMG_SIZE),
        CLAHETransform(clip_limit=2.0),
        transforms.ToTensor()
    ])
    
    # Sample random indices
    indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    
    mean = 0.0
    std = 0.0
    count = 0
    
    for idx in tqdm(indices, desc="Sampling"):
        try:
            if hasattr(dataset, 'df'):
                # CheXpert/NIH datasets
                row = dataset.df.iloc[idx]
                if hasattr(dataset, 'root_dir'):
                    # CheXpert - match CheXpertDataset logic
                    rel_path = row['Path']
                    rel_path_normalized = rel_path.replace('/', os.sep)
                    img_path = os.path.join(dataset.root_dir, rel_path_normalized)
                    
                    # If not found and has prefix, try stripping it
                    if not os.path.exists(img_path) and rel_path.startswith('CheXpert-v1.0-small/'):
                        rel_path_stripped = rel_path.replace('CheXpert-v1.0-small/', '', 1).replace('/', os.sep)
                        img_path = os.path.join(dataset.root_dir, rel_path_stripped)
                else:
                    # NIH
                    img_path = os.path.join(dataset.image_dir, row['Image Index'])
            else:
                # Pneumonia dataset
                img_path = dataset.image_paths[idx]
            
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('L')
                image_tensor = temp_transform(image)
                mean += image_tensor.mean()
                std += image_tensor.std()
                count += 1
        except Exception as e:
            continue
    
    if count > 0:
        mean = mean / count
        std = std / count
        print(f"Dataset stats - Mean: {mean:.4f}, Std: {std:.4f}")
        return mean.item(), std.item()
    else:
        print("Warning: Could not calculate stats, using ImageNet defaults")
        return 0.485, 0.229

def cleanup_temp_files():
    """Clean up temporary files and old checkpoints"""
    print("\nCleaning up temporary files...")
    
    temp_files = ['temp_nih_train.csv', 'temp_nih_val.csv']
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"Cleaned up: {temp_file}")
            except Exception as e:
                print(f"Warning: Could not remove {temp_file}: {e}")
    
    # Keep only latest 2 emergency checkpoints
    try:
        emergency_checkpoints = []
        for file in os.listdir(CHECKPOINT_DIR):
            if file.startswith('emergency_epoch_'):
                emergency_checkpoints.append(os.path.join(CHECKPOINT_DIR, file))
        
        emergency_checkpoints.sort(key=os.path.getctime, reverse=True)
        for checkpoint in emergency_checkpoints[2:]:
            os.remove(checkpoint)
            print(f"Cleaned up old checkpoint: {os.path.basename(checkpoint)}")
    except Exception as e:
        print(f"Warning: Checkpoint cleanup issue: {e}")

# ============================================================================
# MASKED FOCAL LOSS WITH LABEL SMOOTHING (PHASE 4)
# ============================================================================
class MaskedSmoothedFocalLoss(nn.Module):
    """
    Smoothed focal loss with optional masks (ignore uncertain labels).
    masks: (N, C) with 1=certain, 0=ignore
    """
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets, masks=None):
        targets_smoothed = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        bce = F.binary_cross_entropy_with_logits(inputs, targets_smoothed, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce  # (N, C)

        if masks is not None:
            focal = focal * masks
            denom = masks.sum().clamp_min(1e-6)
            return focal.sum() / denom

        return focal.mean()

class SmoothedFocalLoss(nn.Module):
    """Legacy: Focal Loss with multi-label smoothing (kept for compatibility)"""
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        targets_smoothed = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets_smoothed, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class FocalLoss(nn.Module):
    """Original Focal Loss (kept for compatibility)"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def create_stratified_sampled_loader(dataset, sample_size=3000, batch_size=8, num_workers=2):
    """
    Create a DataLoader with stratified sampling to preserve class distribution.
    
    This ensures validation metrics remain reliable across imbalanced medical datasets
    by maintaining the original class proportions in the sampled subset.
    
    Args:
        dataset: PyTorch Dataset with .labels (N x C numpy array) or .labels_list attribute
        sample_size: Number of samples to include (default: 3000)
        batch_size: Batch size for DataLoader (default: 8)
        num_workers: Number of worker processes (default: 2)
        
    Returns:
        DataLoader: Sampled data loader with preserved class distribution
        
    Example:
        >>> val_loader = create_stratified_sampled_loader(val_dataset, sample_size=3000)
        >>> # Class distribution preserved in sampled subset
    """
    total_samples = len(dataset)
    if total_samples <= sample_size:
        # If dataset smaller than sample size, use full dataset
        return create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS
        )
    
    # Get labels for stratification
    if hasattr(dataset, 'labels') and isinstance(dataset.labels, np.ndarray):
        # Multi-label case: use first positive label for stratification
        labels = dataset.labels
        stratify_labels = []
        for label_vec in labels:
            positive_indices = np.where(label_vec == 1)[0]
            if len(positive_indices) > 0:
                stratify_labels.append(positive_indices[0])  # Use first positive label
            else:
                stratify_labels.append(-1)  # No positive labels
    elif hasattr(dataset, 'labels_list'):
        # Single-label case (e.g., Pneumonia dataset)
        stratify_labels = dataset.labels_list
    else:
        # Fallback: random sampling if no labels available
        indices = random.sample(range(total_samples), sample_size)
        subset = torch.utils.data.Subset(dataset, indices)
        return create_dataloader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS
        )
    
    # Perform stratified sampling
    try:
        all_indices = np.arange(total_samples)
        # Use train_test_split for stratified sampling
        sampled_indices, _ = train_test_split(
            all_indices,
            train_size=sample_size,
            stratify=stratify_labels,
            random_state=42
        )
        print(f"Stratified sampling: {len(sampled_indices)} samples (class distribution preserved)")
    except ValueError as e:
        # Fallback to random sampling if stratification fails (e.g., too few samples per class)
        print(f"Stratification failed ({e}), using random sampling instead")
        sampled_indices = random.sample(range(total_samples), sample_size)
    
    subset = torch.utils.data.Subset(dataset, sampled_indices)
    # CRITICAL FIX: Use create_dataloader helper to avoid prefetch_factor=None bug
    return create_dataloader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS
    )

# ============================================================================
# MIXUP
# ============================================================================
def mixup_data(x, y, masks=None, alpha=0.15):
    """Mask-aware mixup for Phase 4"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    if masks is None:
        masks = torch.ones_like(y)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    m_a, m_b = masks, masks[index]
    return mixed_x, y_a, y_b, m_a, m_b, lam

def mixup_criterion_masked(criterion, pred, y_a, y_b, m_a, m_b, lam):
    """Mask-aware mixup criterion for Phase 4"""
    return lam * criterion(pred, y_a, m_a) + (1 - lam) * criterion(pred, y_b, m_b)

# ============================================================================
# ENHANCED TRANSFORMS
# ============================================================================
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        img_np = np.array(img, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        if len(img_np.shape) == 2:
            img_np = clahe.apply(img_np)
        else:
            for i in range(3):
                img_np[:, :, i] = clahe.apply(img_np[:, :, i])
        return Image.fromarray(img_np)

def get_transforms(train=True, enable_clahe=True):
    """
    Get image transforms for training or validation.
    
    Args:
        train: If True, apply training augmentations
        enable_clahe: If True, apply CLAHE (disable for DirectML training speed optimization)
    
    Returns:
        transforms.Compose: Composed transforms
    """
    if train:
        transforms_list = [
            # Resize smaller edge to 340, maintaining aspect ratio
            transforms.Resize(340),
            # Random crop ensures square input without distortion
            transforms.RandomCrop(IMG_SIZE),
        ]
        
        # Only apply CLAHE if enabled (bottleneck on DirectML with NUM_WORKERS=0)
        if enable_clahe:
            transforms_list.append(CLAHETransform(clip_limit=2.0))
        
        transforms_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])  # ImageNet-compatible stats for better transfer
        ])
        
        return transforms.Compose(transforms_list)
    else:
        # Always use CLAHE for validation (fewer images, accurate metrics needed)
        return transforms.Compose([
            transforms.Resize(340),
            # Center crop for validation to preserve anatomy
            transforms.CenterCrop(IMG_SIZE),
            CLAHETransform(clip_limit=2.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])  # ImageNet-compatible stats for better transfer
        ])

def create_sampled_loader(dataset, batch_size, sample_size=3000):
    """
    Create validation loader with random sampling for faster validation.
    DEPRECATED: Use create_stratified_sampled_loader for better class balance.
    Kept for backward compatibility - uses version-safe create_dataloader helper.
    """
    from torch.utils.data import SubsetRandomSampler
    
    if len(dataset) <= sample_size:
        # Use full dataset if smaller than sample size
        return create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS
        )
    
    # Sample random subset
    indices = np.random.choice(len(dataset), size=sample_size, replace=False)
    sampler = SubsetRandomSampler(indices)
    
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS
    )

# ============================================================================
# DATASET CLASSES (Same as phase 2.5)
# ============================================================================
class CheXpertDataset(Dataset):
    """
    CheXpert dataset loader with automatic path handling and label preprocessing.
    
    Handles uncertain labels (-1) by converting to 0, normalizes path separators,
    and provides fallback mechanism for missing images.
    
    Args:
        csv_path: Path to CheXpert CSV file
        root_dir: Root directory containing images
        transform: Optional torchvision transforms
        selected_labels: List of labels to use (default: SELECTED_LABELS)
        sample_size: Optional dataset size limit for debugging
    """
    def __init__(self, csv_path, root_dir, transform=None, selected_labels=None, sample_size=None, return_masks=True):
        self.df = pd.read_csv(csv_path)
        if sample_size and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        self.root_dir = root_dir
        self.transform = transform
        self.selected_labels = selected_labels or SELECTED_LABELS
        self.fallback_count = 0  # Track missing images for debugging
        self.return_masks = return_masks
        
        # CheXpert → NIH disease name mapping
        # CheXpert diseases that match NIH (we'll use these)
        chexpert_to_nih = {
            'Atelectasis': 'Atelectasis',
            'Cardiomegaly': 'Cardiomegaly',
            'Consolidation': 'Consolidation',
            'Edema': 'Edema',
            'Pleural Effusion': 'Effusion',  # CheXpert uses full name, NIH uses short
            'Pneumonia': 'Pneumonia',
            'Pneumothorax': 'Pneumothorax'
        }
        
        # Initialize labels and masks for all 14 diseases
        num_samples = len(self.df)
        self.labels = np.zeros((num_samples, len(self.selected_labels)), dtype=np.float32)
        self.masks = np.ones((num_samples, len(self.selected_labels)), dtype=np.float32)  # Default: all certain
        
        # Process only diseases that exist in CheXpert CSV
        for chexpert_name, nih_name in chexpert_to_nih.items():
            if chexpert_name in self.df.columns and nih_name in self.selected_labels:
                nih_idx = LABEL_TO_IDX[nih_name]
                col_data = self.df[chexpert_name].to_numpy(dtype=np.float32)
                
                # Handle NaN and -1
                col_data = np.nan_to_num(col_data, nan=0.0)  # NaN → 0 (negative)
                self.masks[:, nih_idx] = (col_data != -1.0).astype(np.float32)  # -1 → masked
                self.labels[:, nih_idx] = np.where(col_data == -1.0, 0.0, col_data)
        
        # For NIH diseases NOT in CheXpert (Emphysema, Fibrosis, etc.):
        # - Labels remain 0 (no positive examples)
        # - Masks remain 1 (treat as certain negatives)
        # This is correct: CheXpert simply doesn't have these diseases labeled
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            # Robust path resolution: try with and without prefix stripping
            rel_path = row['Path']
            rel_path_normalized = rel_path.replace('/', os.sep)
            
            # Try with original path first
            img_path = os.path.join(self.root_dir, rel_path_normalized)
            
            # If not found and has prefix, try stripping it
            if not os.path.exists(img_path) and rel_path.startswith('CheXpert-v1.0-small/'):
                rel_path_stripped = rel_path.replace('CheXpert-v1.0-small/', '', 1).replace('/', os.sep)
                img_path = os.path.join(self.root_dir, rel_path_stripped)
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            image = Image.open(img_path).convert('L')
            
            if self.transform:
                image = self.transform(image)
            
            # SPEEDUP FIX: Use precomputed labels (same as NIH dataset)
            labels = torch.from_numpy(self.labels[idx]).float()
            
            if self.return_masks:
                mask = torch.from_numpy(self.masks[idx]).float()
                return image, labels, mask
            else:
                return image, labels
            
        except Exception as e:
            # Return blank image as fallback - track for debugging
            self.fallback_count += 1
            if self.fallback_count <= 10:  # Log first 10 only to avoid spam
                print(f"Warning: CheXpert image {idx} missing/corrupt: {e}")
            image = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
            if self.transform:
                image = self.transform(image)
            labels = torch.zeros(len(self.selected_labels), dtype=torch.float32)
            if self.return_masks:
                mask = torch.zeros(len(self.selected_labels), dtype=torch.float32)  # IGNORE blank
                return image, labels, mask
            return image, labels

class NIHDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, selected_labels=None, sample_size=None, patient_level_split=True, return_masks=True):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.selected_labels = selected_labels or SELECTED_LABELS
        self.fallback_count = 0  # Track missing images
        self.return_masks = return_masks
        
        if patient_level_split:
            unique_patients = self.df['Patient ID'].unique()
            if sample_size and sample_size < len(unique_patients):
                sampled_patients = np.random.choice(unique_patients, size=sample_size, replace=False)
                self.df = self.df[self.df['Patient ID'].isin(sampled_patients)].reset_index(drop=True)
        
        # CRITICAL: Precompute label matrix for all 14 diseases
        self.df['Finding Labels'] = self.df['Finding Labels'].fillna('')
        self.labels = np.zeros((len(self.df), len(self.selected_labels)), dtype=np.float32)
        
        def has(lbl):
            # Exact match in pipe-separated list
            return self.df['Finding Labels'].str.contains(rf'(^|\|){lbl}(\||$)', regex=True)
        
        # Map all 14 NIH diseases with correct CSV names
        # CRITICAL: NIH CSV uses "Pleural Thickening" (space), not "Pleural_Thickening" (underscore)
        NIH_NAME_MAP = {
            "Pleural_Thickening": "Pleural Thickening"  # Map internal name to CSV name
        }
        
        # Map all diseases dynamically
        for disease in self.selected_labels:
            csv_name = NIH_NAME_MAP.get(disease, disease)  # Use mapped name or original
            self.labels[:, LABEL_TO_IDX[disease]] = has(csv_name).astype(np.float32).values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_path = os.path.join(self.image_dir, row['Image Index'])
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            image = Image.open(img_path).convert('L')
            
            if self.transform:
                image = self.transform(image)
            
            # Use precomputed labels (much faster + consistent)
            labels = torch.from_numpy(self.labels[idx]).float()
            
            if self.return_masks:
                # NIH has no uncertain labels - all certain
                mask = torch.ones(len(self.selected_labels), dtype=torch.float32)
                return image, labels, mask
            else:
                return image, labels
            
        except Exception as e:
            # Return blank image as fallback
            self.fallback_count += 1
            if self.fallback_count <= 10:
                print(f"Warning: NIH image {idx} missing/corrupt: {e}")
            image = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
            if self.transform:
                image = self.transform(image)
            labels = torch.zeros(len(self.selected_labels), dtype=torch.float32)
            if self.return_masks:
                mask = torch.zeros(len(self.selected_labels), dtype=torch.float32)  # IGNORE blank
                return image, labels, mask
            return image, labels

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, return_masks=True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.fallback_count = 0  # Track missing images
        self.return_masks = return_masks
        
        split_dir = os.path.join(root_dir, split)
        self.image_paths = []
        self.labels_list = []
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels_list.append(1.0 if class_name == 'PNEUMONIA' else 0.0)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            image = Image.open(img_path).convert('L')
            
            if self.transform:
                image = self.transform(image)
            
            # Pneumonia dataset: Map binary label to Pneumonia disease (index 12)
            # In NIH terminology, Pneumonia is its own disease class
            labels = torch.zeros(len(SELECTED_LABELS), dtype=torch.float32)
            if 'Pneumonia' in LABEL_TO_IDX:
                pneumonia_idx = LABEL_TO_IDX['Pneumonia']
                labels[pneumonia_idx] = self.labels_list[idx]
            else:
                raise ValueError("Pneumonia not in SELECTED_LABELS! Check configuration.")
            
            if self.return_masks:
                # Pneumonia has no uncertain labels - all certain
                mask = torch.ones(len(SELECTED_LABELS), dtype=torch.float32)
                return image, labels, mask
            else:
                return image, labels
            
        except Exception as e:
            # Return blank image as fallback
            self.fallback_count += 1
            if self.fallback_count <= 10:
                print(f"Warning: Pneumonia image {idx} missing/corrupt: {e}")
            image = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
            if self.transform:
                image = self.transform(image)
            labels = torch.zeros(len(SELECTED_LABELS), dtype=torch.float32)
            if self.return_masks:
                mask = torch.zeros(len(SELECTED_LABELS), dtype=torch.float32)  # IGNORE blank
                return image, labels, mask
            return image, labels

# ============================================================================
# MODEL
# ============================================================================
class DenseNet121(nn.Module):
    def __init__(self, num_classes=5, grayscale=True, use_gradient_checkpointing=False):
        super().__init__()
        # Version-safe ImageNet pretrained weights (supports PyTorch 1.9+)
        try:
            densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        except (AttributeError, Exception):
            # Fallback for older torchvision (< 0.13)
            densenet = models.densenet121(pretrained=True)
        
        if grayscale:
            # Adapt first conv for grayscale by averaging RGB weights
            # IMPORTANT: Modify in-place to preserve layer naming for checkpoint compatibility
            original_conv = densenet.features.conv0
            adapted_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Average RGB channels to create grayscale initialization
            with torch.no_grad():
                adapted_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            densenet.features.conv0 = adapted_conv
        
        # Keep original features structure to maintain checkpoint compatibility
        self.features = densenet.features
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        num_ftrs = 1024
        # Enhanced classifier with dual dropout for better regularization
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),  # Use config value (0.25 for Phase 4)
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT * 0.6),  # Proportional second dropout
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        if self.use_gradient_checkpointing and self.training:
            # Modern PyTorch: use non-reentrant checkpointing (safer, fewer edge cases)
            try:
                x = torch.utils.checkpoint.checkpoint_sequential(
                    self.features,
                    4,
                    x,
                    use_reentrant=False  # Modern PyTorch 1.11+ recommended mode
                )
            except TypeError:
                # Fallback for older PyTorch (< 1.11): use reentrant mode
                if not x.requires_grad:
                    x = x.requires_grad_(True)
                x = checkpoint_sequential(self.features, 4, x)
        else:
            x = self.features(x)
        
        # Important: apply activation consistently for both paths
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_model(num_classes=5):
    """
    Create DenseNet-121 model with optional checkpoint loading and parameter tracking.
    
    Features:
        - Grayscale input adaptation via channel averaging
        - Optional gradient checkpointing for memory efficiency
        - Enhanced classifier with dual dropout
        - Automatic checkpoint resume with state restoration
        - Version-safe initialization (PyTorch 1.9+, torchvision 0.10+)
        
    Args:
        num_classes: Number of output classes (default: 5)
        
    Returns:
        tuple: (model, resume_info_dict)
            resume_info contains: start_epoch, best_auc, optimizer_state, scheduler_state
    """
    model = DenseNet121(
        num_classes=num_classes, 
        grayscale=True,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING
    )
    
    # Print model parameter info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Memory: ~{(total_params * 4) / 1e6:.1f}MB (fp32)\n")
    
    resume_info = {'start_epoch': 1, 'best_auc': 0.0, 'optimizer_state': None, 'scheduler_state': None}
    
    if RESUME_FROM_BEST:
        if not os.path.exists(BEST_CHECKPOINT):
            print(f"ERROR: Checkpoint not found at {BEST_CHECKPOINT}")
            print(f"Set RESUME_FROM_BEST=False to train from scratch, or provide valid checkpoint.")
            sys.exit(1)
        
        print(f"Loading best checkpoint from: {BEST_CHECKPOINT}")
        # Version-safe checkpoint load (PyTorch 1.13+ uses weights_only, older versions don't)
        try:
            checkpoint = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        except TypeError:
            # Fallback for PyTorch < 1.13
            checkpoint = torch.load(BEST_CHECKPOINT, map_location="cpu")
        
        state = checkpoint['model_state_dict']
        
        # Check if checkpoint is 5-class or 14-class
        checkpoint_classes = None
        if 'classifier.4.weight' in state:
            checkpoint_classes = state['classifier.4.weight'].shape[0]
            print(f"Checkpoint has {checkpoint_classes} output classes")
        
        if checkpoint_classes == num_classes:
            # Same number of classes - load full model
            model.load_state_dict(state)
            print(f"Loaded full model (same {num_classes} classes)")
        else:
            # Different number of classes - load backbone only
            filtered = {k: v for k, v in state.items() if not k.startswith('classifier.')}
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            print(f"Loaded backbone only (checkpoint: {checkpoint_classes} classes -> model: {num_classes} classes)")
            print(f"Classifier re-initialized for {num_classes} classes")
            if missing:
                print(f"  Missing keys: {len(missing)} items")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)} items")
        
        # Resume training state for continuity
        # RESET start_epoch to 1 for Phase 4 bookkeeping
        resume_info['start_epoch'] = 1  # Fresh start for Phase 4
        resume_info['best_auc'] = 0.0   # Fresh best AUC for 14-class model
        resume_info['optimizer_state'] = None  # Don't resume optimizer
        resume_info['scheduler_state'] = None  # Don't resume scheduler
        
        # Safe formatting with default values
        epoch = checkpoint.get('epoch', 'Unknown')
        mean_auc = checkpoint.get('mean_auc', None)
        nih_auc = checkpoint.get('nih_auc', None)
        chexpert_auc = checkpoint.get('chexpert_auc', None)
        pneumonia_auc = checkpoint.get('pneumonia_auc', None)
        
        print(f"Resumed from Epoch {epoch}")
        if mean_auc is not None:
            print(f"  Mean AUC: {mean_auc:.4f}")
            if nih_auc is not None:
                print(f"  NIH AUC: {nih_auc:.4f}")
            if chexpert_auc is not None:
                print(f"  CheXpert AUC: {chexpert_auc:.4f}")
            if pneumonia_auc is not None:
                print(f"  Pneumonia AUC: {pneumonia_auc:.4f}")
        else:
            print("  Checkpoint loaded successfully (metrics not available)")
        print(f"  Resuming training from epoch {resume_info['start_epoch']} with best AUC {resume_info['best_auc']:.4f}")
    else:
        print("Initializing from ImageNet pretrained weights")
    
    # Reapply channels_last format after resume (critical for CUDA performance)
    if str(DEVICE) == "cuda":
        model = model.to(memory_format=torch.channels_last)
        print("Applied channels-last memory format post-resume")
    
    return model, resume_info

# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, epoch, start_epoch, end_epoch):
    """
    Train model for one epoch with gradient accumulation and mixed precision.
    
    Args:
        model: PyTorch model to train
        loader: DataLoader for training data
        criterion: Loss function (e.g., SmoothedFocalLoss)
        optimizer: Optimizer (e.g., AdamW)
        scaler: GradScaler for mixed precision (None for CPU/DirectML)
        scheduler: Learning rate scheduler (stepped per batch for smooth warm restarts)
        epoch: Current epoch number (for logging)
        start_epoch: Starting epoch number for Phase 4
        end_epoch: Ending epoch number for Phase 4
        
    Returns:
        float: Average training loss for the epoch
        
    Training Features:
        - Gradient accumulation (effective batch = BATCH_SIZE × ACCUMULATION_STEPS)
        - Mixed precision training (AMP) on CUDA devices
        - Gradient clipping (max_norm=1.0) to prevent exploding gradients
        - Per-batch scheduler stepping for CosineAnnealingWarmRestarts
        - Handles -1 (uncertain) labels properly
        - Progress logging every 100 batches
    """
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
    
    num_batches = len(loader)
    remainder = num_batches % ACCUMULATION_STEPS
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{end_epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Unpack with optional masks (3-tuple for Phase 4)
        if len(batch) == 3:
            images, labels, masks = batch
            use_non_blocking = (str(DEVICE) == "cuda" and PIN_MEMORY)
            masks = masks.to(DEVICE, non_blocking=use_non_blocking)
        else:
            # Fallback for 2-tuple (shouldn't happen in Phase 4)
            images, labels = batch
            masks = None
        
        # non_blocking only safe on CUDA with pin_memory
        use_non_blocking = (str(DEVICE) == "cuda" and PIN_MEMORY)
        images = images.to(DEVICE, non_blocking=use_non_blocking)
        labels = labels.to(DEVICE, non_blocking=use_non_blocking)
        
        # Use channels-last format on CUDA for better memory bandwidth
        if str(DEVICE) == "cuda" and images.dim() == 4:
            images = images.to(memory_format=torch.channels_last)
        
        # Correct scaling for final partial accumulation window
        if remainder != 0 and batch_idx >= (num_batches - remainder):
            accum_in_window = remainder
        else:
            accum_in_window = ACCUMULATION_STEPS
        
        # MixUp augmentation (mask-aware)
        if MIXUP_ALPHA > 0 and random.random() < 0.5:
            images, labels_a, labels_b, masks_a, masks_b, lam = mixup_data(images, labels, masks, MIXUP_ALPHA)
            do_mixup = True
        else:
            labels_a, labels_b, masks_a, masks_b, lam = labels, None, masks, None, 1.0
            do_mixup = False
        
        # Forward pass with AMP
        if AMP_ENABLED and scaler:
            with autocast():
                outputs = model(images)
                if do_mixup:
                    loss = mixup_criterion_masked(criterion, outputs, labels_a, labels_b, masks_a, masks_b, lam)
                else:
                    loss = criterion(outputs, labels_a, masks)
                loss = loss / accum_in_window  # Use correct window size
            
            scaler.scale(loss).backward()
            
            # FIX: Step if accumulation limit reached OR if this is the very last batch
            if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or ((batch_idx + 1) == num_batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                
                # Track scale before step to detect gradient overflow
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Only step scheduler if optimizer actually stepped (no overflow)
                if scaler.get_scale() >= scale_before:
                    ep_rel = (epoch - start_epoch)
                    scheduler.step(ep_rel + (batch_idx + 1) / num_batches)
        else:
            outputs = model(images)
            if do_mixup:
                loss = mixup_criterion_masked(criterion, outputs, labels_a, labels_b, masks_a, masks_b, lam)
            else:
                loss = criterion(outputs, labels_a, masks)  # Pass masks
            loss = loss / accum_in_window  # Use correct window size
            
            loss.backward()
            
            # FIX: Step if accumulation limit reached OR if this is the very last batch
            if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or ((batch_idx + 1) == num_batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Step scheduler per batch with fractional epoch for smooth warm restarts (finetune-relative)
                ep_rel = (epoch - start_epoch)
                scheduler.step(ep_rel + (batch_idx + 1) / num_batches)
        
        # Recover original-scale loss for logging
        running_loss += loss.item() * accum_in_window
        
        # Enhanced progress monitoring every 100 batches
        if batch_idx % 100 == 0:
            progress = (batch_idx / num_batches) * 100
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = running_loss / (batch_idx + 1)
            print(f"  Batch {batch_idx}/{num_batches} ({progress:.1f}%) | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
            if str(DEVICE) == "cuda":
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"  GPU Memory: {allocated:.1f}GB")
        
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
    
    return running_loss / num_batches

def validate(model, loader, dataset_name=""):
    """
    Validate model on given dataset with MASK-AWARE AUC computation.
    Only certain labels (mask=1) contribute to metrics.
    """
    was_training = model.training
    model.eval()
    all_preds = []
    all_labels = []
    all_masks = []
    
    print(f"Validating on {dataset_name} ({len(loader.dataset)} samples)...")
    
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Validating {dataset_name}"):
            # Handle 3-tuple or 2-tuple
            if len(batch) == 3:
                images, labels, masks = batch
            else:
                images, labels = batch
                masks = torch.ones_like(labels)  # All certain for NIH/Pneumonia
            
            use_non_blocking = (str(DEVICE) == "cuda" and PIN_MEMORY)
            images = images.to(DEVICE, non_blocking=use_non_blocking)
            
            # Use channels-last format on CUDA for consistency with training
            if str(DEVICE) == "cuda" and images.dim() == 4:
                images = images.to(memory_format=torch.channels_last)
            
            # Test-Time Augmentation (optional, ~2x slower but can improve AUC)
            if USE_TTA:
                outputs = (model(images) + model(torch.flip(images, [3]))) / 2
            else:
                outputs = model(images)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_preds.append(probs)
            all_labels.append(labels.numpy())
            all_masks.append(masks.numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_masks = np.vstack(all_masks)
    
    # Compute AUC per disease (only on certain labels)
    disease_aucs = []
    per_class_aucs = {}
    valid_diseases = 0
    
    for i, label in enumerate(SELECTED_LABELS):
        # Only use samples where mask=1 (certain labels)
        keep = all_masks[:, i] == 1
        y_true = all_labels[keep, i]
        y_pred = all_preds[keep, i]
        
        if len(y_true) > 0 and len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred)
            disease_aucs.append(auc)
            per_class_aucs[label] = auc
            valid_diseases += 1
            print(f"  {label:20s} {auc:.4f} ({keep.sum()}/{len(keep)} certain)")
        else:
            per_class_aucs[label] = None
            print(f"  {label:20s} N/A (single class or no certain labels)")
    
    mean_auc = float(np.mean(disease_aucs)) if disease_aucs else 0.0
    print(f"  {'Mean AUC':20s} {mean_auc:.4f} (across {valid_diseases} classes)\n")
    
    # Restore original training state instead of forcing train mode
    if was_training:
        model.train()
    return mean_auc, per_class_aucs

def compute_min_auc(per_class_dicts):
    """
    Returns min AUC across all provided per-class AUC dicts,
    ignoring None / NaN values (classes with single-class labels in that val set).
    
    This enforces "85%+ on ALL diseases" by optimizing the worst-case class.
    
    Args:
        per_class_dicts: List of per-class AUC dictionaries from validate()
        
    Returns:
        float: Minimum AUC across all valid classes, or 0.0 if no valid classes
    """
    vals = []
    for d in per_class_dicts:
        if not d:
            continue
        for v in d.values():
            if v is None:
                continue
            v = float(v)
            if not np.isnan(v):
                vals.append(v)
    return min(vals) if vals else 0.0

# ============================================================================
# MAIN
# ============================================================================
def main():
    try:
        print("\n" + "="*70)
        print("PHASE 4: MASKED FINE-TUNING")
        print("="*70)
        
        # Validate all paths exist
        validate_paths()
        
        # Print system information
        print_system_info()
        
        # Save training configuration
        save_training_config()
        
        # Load datasets with optimized data loading
        print("\nLoading datasets...")
        
        # Disable CLAHE for DirectML training (3-4x speedup, bottleneck with NUM_WORKERS=0)
        enable_train_clahe = not IS_DIRECTML
        if not enable_train_clahe:
            print(" CLAHE disabled for training (DirectML optimization)")
            print("   Validation still uses CLAHE for accurate metrics\n")
        
        train_transform = get_transforms(train=True, enable_clahe=enable_train_clahe)
        val_transform = get_transforms(train=False, enable_clahe=True)
        
        # ============================================================================
        # OPTIMIZED SPLITS: NIH 80/10/10 (Stratified), CheXpert 80/10/10, Pneumonia 70/15/15
        # ============================================================================
        print("\n" + "="*70)
        print("CREATING OPTIMIZED TRAIN/VAL/TEST SPLITS")
        print("  NIH:       80/10/10 (Stratified patient-level)")
        print("  CheXpert:  80/10/10 (Patient-level)")
        print("  Pneumonia: 70/15/15 (Stratified class-balanced)")
        print("="*70)
        
        # --- CheXpert 80/10/10 Split ---
        chexpert_fixed_train = os.path.join(CHECKPOINT_DIR, "chexpert_fixed_train.csv")
        chexpert_fixed_val = os.path.join(CHECKPOINT_DIR, "chexpert_fixed_val.csv")
        chexpert_fixed_test = os.path.join(CHECKPOINT_DIR, "chexpert_fixed_test.csv")
        
        if os.path.exists(chexpert_fixed_train) and os.path.exists(chexpert_fixed_val) and os.path.exists(chexpert_fixed_test):
            print(f"\n[OK] Loading existing CheXpert 80/10/10 splits...")
            chexpert_train_df = pd.read_csv(chexpert_fixed_train)
            chexpert_val_df = pd.read_csv(chexpert_fixed_val)
            chexpert_test_df = pd.read_csv(chexpert_fixed_test)
        else:
            print(f"\n[NEW] Creating NEW CheXpert 80/10/10 splits (patient-level)...")
            chexpert_df = pd.read_csv(CHEXPERT_TRAIN_CSV)
            
            # Extract patient IDs from Path column (format: patient12345/study1/...)
            chexpert_df['Patient'] = chexpert_df['Path'].str.extract(r'(patient\d+)')
            patients = chexpert_df['Patient'].unique()
            
            # Shuffle and split patients 80/10/10
            np.random.seed(42)
            np.random.shuffle(patients)
            
            n_patients = len(patients)
            train_end = int(0.8 * n_patients)
            val_end = int(0.9 * n_patients)
            
            train_patients = set(patients[:train_end])
            val_patients = set(patients[train_end:val_end])
            test_patients = set(patients[val_end:])
            
            chexpert_train_df = chexpert_df[chexpert_df['Patient'].isin(train_patients)].copy()
            chexpert_val_df = chexpert_df[chexpert_df['Patient'].isin(val_patients)].copy()
            chexpert_test_df = chexpert_df[chexpert_df['Patient'].isin(test_patients)].copy()
            
            # Save fixed splits
            chexpert_train_df.to_csv(chexpert_fixed_train, index=False)
            chexpert_val_df.to_csv(chexpert_fixed_val, index=False)
            chexpert_test_df.to_csv(chexpert_fixed_test, index=False)
            print(f"   Saved to: {chexpert_fixed_train}")
        
        print(f"   CheXpert - Train: {len(chexpert_train_df):,} | Val: {len(chexpert_val_df):,} | Test: {len(chexpert_test_df):,}")
        
        # Create CheXpert datasets
        chexpert_train_df.to_csv('temp_chexpert_train.csv', index=False)
        chexpert_val_df.to_csv('temp_chexpert_val.csv', index=False)
        chexpert_test_df.to_csv('temp_chexpert_test.csv', index=False)
        
        chexpert_train = CheXpertDataset('temp_chexpert_train.csv', CHEXPERT_ROOT, train_transform, SELECTED_LABELS, CHEXPERT_SAMPLE_SIZE)
        chexpert_valid = CheXpertDataset('temp_chexpert_val.csv', CHEXPERT_ROOT, val_transform, SELECTED_LABELS)
        chexpert_test = CheXpertDataset('temp_chexpert_test.csv', CHEXPERT_ROOT, val_transform, SELECTED_LABELS)
        
        # --- NIH 80/10/10 Split (STRATIFIED) ---
        nih_fixed_train = os.path.join(CHECKPOINT_DIR, "nih_fixed_train.csv")
        nih_fixed_val = os.path.join(CHECKPOINT_DIR, "nih_fixed_val.csv")
        nih_fixed_test = os.path.join(CHECKPOINT_DIR, "nih_fixed_test.csv")
        
        if os.path.exists(nih_fixed_train) and os.path.exists(nih_fixed_val) and os.path.exists(nih_fixed_test):
            print(f"\n[OK] Loading existing NIH 80/10/10 stratified splits...")
            nih_train_df = pd.read_csv(nih_fixed_train)
            nih_val_df = pd.read_csv(nih_fixed_val)
            nih_test_df = pd.read_csv(nih_fixed_test)
        else:
            print(f"\n[NEW] Creating NEW NIH 80/10/10 STRATIFIED splits (patient-level)...")
            print("   Strategy: Ensure minimum 30 val/test samples for rare diseases (Hernia, Pneumonia, Fibrosis)")
            nih_df = pd.read_csv(NIH_CSV_PATH)
            
            # Patient-level stratified split
            patients = nih_df['Patient ID'].unique()
            np.random.seed(42)
            np.random.shuffle(patients)
            
            # Standard 80/10/10 split on patients
            n_patients = len(patients)
            train_end = int(0.8 * n_patients)
            val_end = int(0.9 * n_patients)
            
            train_patients = set(patients[:train_end])
            val_patients = set(patients[train_end:val_end])
            test_patients = set(patients[val_end:])
            
            nih_train_df = nih_df[nih_df['Patient ID'].isin(train_patients)].copy()
            nih_val_df = nih_df[nih_df['Patient ID'].isin(val_patients)].copy()
            nih_test_df = nih_df[nih_df['Patient ID'].isin(test_patients)].copy()
            
            # Verify minimum samples for rare diseases
            rare_diseases = ['Hernia', 'Pneumonia', 'Fibrosis']
            for disease in rare_diseases:
                # Count samples containing this disease in val/test
                val_count = nih_val_df['Finding Labels'].str.contains(disease, na=False).sum()
                test_count = nih_test_df['Finding Labels'].str.contains(disease, na=False).sum()
                print(f"   {disease}: Val={val_count}, Test={test_count} samples")
                if val_count < 30 or test_count < 30:
                    print(f"   [WARNING] {disease} has <30 samples in val or test (acceptable for rare disease)")
            
            # Save fixed splits
            nih_train_df.to_csv(nih_fixed_train, index=False)
            nih_val_df.to_csv(nih_fixed_val, index=False)
            nih_test_df.to_csv(nih_fixed_test, index=False)
            print(f"   Saved to: {nih_fixed_train}")
        
        print(f"   NIH      - Train: {len(nih_train_df):,} | Val: {len(nih_val_df):,} | Test: {len(nih_test_df):,}")
        
        if NIH_SAMPLE_SIZE:
            train_patients = nih_train_df['Patient ID'].unique()
            num_patients_to_sample = min(NIH_SAMPLE_SIZE, len(train_patients))
            sampled_patients = np.random.choice(train_patients, num_patients_to_sample, replace=False)
            nih_train_df = nih_train_df[nih_train_df['Patient ID'].isin(sampled_patients)]
        
        # Save to temp for Dataset class
        nih_train_df.to_csv('temp_nih_train.csv', index=False)
        nih_val_df.to_csv('temp_nih_val.csv', index=False)
        nih_test_df.to_csv('temp_nih_test.csv', index=False)
        
        nih_train = NIHDataset('temp_nih_train.csv', NIH_IMAGE_DIR, train_transform, SELECTED_LABELS)
        nih_valid = NIHDataset('temp_nih_val.csv', NIH_IMAGE_DIR, val_transform, SELECTED_LABELS)
        nih_test = NIHDataset('temp_nih_test.csv', NIH_IMAGE_DIR, val_transform, SELECTED_LABELS)
        
        # --- Pneumonia 70/15/15 Split (OPTIMIZED) ---
        pneumonia_fixed_train = os.path.join(CHECKPOINT_DIR, "pneumonia_fixed_train.csv")
        pneumonia_fixed_val = os.path.join(CHECKPOINT_DIR, "pneumonia_fixed_val.csv")
        pneumonia_fixed_test = os.path.join(CHECKPOINT_DIR, "pneumonia_fixed_test.csv")
        
        if os.path.exists(pneumonia_fixed_train) and os.path.exists(pneumonia_fixed_val) and os.path.exists(pneumonia_fixed_test):
            print(f"\n[OK] Loading existing Pneumonia 70/15/15 splits...")
            pneu_train_paths = pd.read_csv(pneumonia_fixed_train)['image_path'].tolist()
            pneu_val_paths = pd.read_csv(pneumonia_fixed_val)['image_path'].tolist()
            pneu_test_paths = pd.read_csv(pneumonia_fixed_test)['image_path'].tolist()
        else:
            print(f"\n[NEW] Creating NEW Pneumonia 70/15/15 splits (stratified)...")
            print("   Strategy: Larger val/test (15% each) for small imbalanced dataset")
            # Collect all images from train + test folders
            all_images = []
            for split_name in ['train', 'test']:
                split_dir = os.path.join(PNEUMONIA_ROOT, split_name)
                for class_name in ['NORMAL', 'PNEUMONIA']:
                    class_dir = os.path.join(split_dir, class_name)
                    if os.path.exists(class_dir):
                        for img_name in os.listdir(class_dir):
                            if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                                img_path = os.path.join(class_dir, img_name)
                                label = 1.0 if class_name == 'PNEUMONIA' else 0.0
                                all_images.append({'image_path': img_path, 'label': label})
            
            # Stratified shuffle and split 70/15/15 (class-balanced)
            from sklearn.model_selection import train_test_split
            
            # Separate by class for stratified sampling
            normal_images = [x for x in all_images if x['label'] == 0.0]
            pneumonia_images = [x for x in all_images if x['label'] == 1.0]
            
            print(f"   Total: {len(all_images)} images (NORMAL: {len(normal_images)}, PNEUMONIA: {len(pneumonia_images)})")
            print(f"   Class ratio: {len(pneumonia_images)/len(normal_images):.2f}:1 (PNEUMONIA:NORMAL)")
            
            # Shuffle both classes
            np.random.seed(42)
            np.random.shuffle(normal_images)
            np.random.shuffle(pneumonia_images)
            
            # Split 70/15/15 for each class
            n_normal = len(normal_images)
            n_pneumonia = len(pneumonia_images)
            
            train_end_normal = int(0.70 * n_normal)
            val_end_normal = int(0.85 * n_normal)
            train_end_pneumonia = int(0.70 * n_pneumonia)
            val_end_pneumonia = int(0.85 * n_pneumonia)
            
            # Combine stratified splits
            pneu_train_list = normal_images[:train_end_normal] + pneumonia_images[:train_end_pneumonia]
            pneu_val_list = normal_images[train_end_normal:val_end_normal] + pneumonia_images[train_end_pneumonia:val_end_pneumonia]
            pneu_test_list = normal_images[val_end_normal:] + pneumonia_images[val_end_pneumonia:]
            
            # Shuffle combined lists
            np.random.shuffle(pneu_train_list)
            np.random.shuffle(pneu_val_list)
            np.random.shuffle(pneu_test_list)
            
            # Print stratified counts
            train_normal = sum(1 for x in pneu_train_list if x['label'] == 0.0)
            val_normal = sum(1 for x in pneu_val_list if x['label'] == 0.0)
            test_normal = sum(1 for x in pneu_test_list if x['label'] == 0.0)
            print(f"   Train: {len(pneu_train_list)} (NORMAL: {train_normal}, PNEUMONIA: {len(pneu_train_list)-train_normal})")
            print(f"   Val:   {len(pneu_val_list)} (NORMAL: {val_normal}, PNEUMONIA: {len(pneu_val_list)-val_normal})")
            print(f"   Test:  {len(pneu_test_list)} (NORMAL: {test_normal}, PNEUMONIA: {len(pneu_test_list)-test_normal})")
            
            # Save as CSV
            pd.DataFrame(pneu_train_list).to_csv(pneumonia_fixed_train, index=False)
            pd.DataFrame(pneu_val_list).to_csv(pneumonia_fixed_val, index=False)
            pd.DataFrame(pneu_test_list).to_csv(pneumonia_fixed_test, index=False)
            print(f"   Saved to: {pneumonia_fixed_train}")
            
            pneu_train_paths = [x['image_path'] for x in pneu_train_list]
            pneu_val_paths = [x['image_path'] for x in pneu_val_list]
            pneu_test_paths = [x['image_path'] for x in pneu_test_list]
        
        print(f"   Pneumonia - Train: {len(pneu_train_paths):,} | Val: {len(pneu_val_paths):,} | Test: {len(pneu_test_paths):,}")
        
        # Create custom Pneumonia datasets from fixed paths
        class PneumoniaCustomDataset(Dataset):
            def __init__(self, image_paths, labels, transform=None, return_masks=True):
                self.image_paths = image_paths
                self.labels_list = labels
                self.transform = transform
                self.return_masks = return_masks
                self.fallback_count = 0
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                try:
                    img_path = self.image_paths[idx]
                    image = Image.open(img_path).convert('L')
                    if self.transform:
                        image = self.transform(image)
                    
                    labels = torch.zeros(len(SELECTED_LABELS), dtype=torch.float32)
                    pneumonia_idx = LABEL_TO_IDX.get('Pneumonia', -1)
                    if pneumonia_idx >= 0:
                        labels[pneumonia_idx] = self.labels_list[idx]
                    
                    if self.return_masks:
                        mask = torch.ones(len(SELECTED_LABELS), dtype=torch.float32)
                        return image, labels, mask
                    return image, labels
                except Exception as e:
                    self.fallback_count += 1
                    if self.fallback_count <= 10:
                        print(f"Warning: Pneumonia image {idx} error: {e}")
                    image = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
                    if self.transform:
                        image = self.transform(image)
                    labels = torch.zeros(len(SELECTED_LABELS), dtype=torch.float32)
                    if self.return_masks:
                        mask = torch.zeros(len(SELECTED_LABELS), dtype=torch.float32)
                        return image, labels, mask
                    return image, labels
        
        pneu_train_df = pd.read_csv(pneumonia_fixed_train)
        pneu_val_df = pd.read_csv(pneumonia_fixed_val)
        pneu_test_df = pd.read_csv(pneumonia_fixed_test)
        
        pneumonia_train = PneumoniaCustomDataset(pneu_train_df['image_path'].tolist(), pneu_train_df['label'].tolist(), train_transform)
        pneumonia_valid = PneumoniaCustomDataset(pneu_val_df['image_path'].tolist(), pneu_val_df['label'].tolist(), val_transform)
        pneumonia_test = PneumoniaCustomDataset(pneu_test_df['image_path'].tolist(), pneu_test_df['label'].tolist(), val_transform)
        
        print("\n" + "="*70)
        print("80/10/10 SPLITS CREATED & PERSISTED")
        print("="*70 + "\n")
        
        # Combine training sets
        train_dataset = ConcatDataset([chexpert_train, nih_train, pneumonia_train])
        
        print(f"\nTotal Training: {len(train_dataset)} images")
        print(f"  - CheXpert: {len(chexpert_train)}")
        print(f"  - NIH: {len(nih_train)}")
        print(f"  - Pneumonia: {len(pneumonia_train)}")
        
        # Dataset-balanced sampling with Pneumonia cap to prevent overfitting
        # Problem: Pneumonia is tiny (~5k), equal weighting causes massive repetition
        # Solution: Cap Pneumonia at 10% contribution, split rest between CheXpert/NIH
        dataset_sizes = [len(chexpert_train), len(nih_train), len(pneumonia_train)]
        
        # Dataset contribution probabilities (tune these, must sum to 1.0)
        # 45% CheXpert, 45% NIH, 10% Pneumonia prevents Pneumonia overfit
        p_chex, p_nih, p_pneu = 0.45, 0.45, 0.10
        
        # Create per-sample weights based on dataset probabilities
        weights = (
            [p_chex / dataset_sizes[0]] * dataset_sizes[0] +  # CheXpert samples
            [p_nih  / dataset_sizes[1]] * dataset_sizes[1] +  # NIH samples
            [p_pneu / dataset_sizes[2]] * dataset_sizes[2]    # Pneumonia samples
        )
        weights = torch.DoubleTensor(weights)
        
        # CRITICAL FIX: Define epoch length by optimizer steps (not dataset size)
        # This makes training progress predictable and validation cadence reasonable
        # CPU is much slower, so use fewer steps per epoch for practical training time
        # DirectML/GPU: 600-1500 steps, CPU: 200 steps for ~2 hour epochs
        if IS_DIRECTML:
            TARGET_OPT_STEPS_PER_EPOCH = 600
        elif str(DEVICE) == "cuda":
            TARGET_OPT_STEPS_PER_EPOCH = 1500
        else:  # CPU
            TARGET_OPT_STEPS_PER_EPOCH = 200  # CPU-optimized: ~2 hours/epoch instead of 15 hours
        
        # Calculate samples needed for target optimizer steps
        # Each optimizer step processes (BATCH_SIZE * ACCUMULATION_STEPS) samples
        num_samples_per_epoch = TARGET_OPT_STEPS_PER_EPOCH * BATCH_SIZE * ACCUMULATION_STEPS
        
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples_per_epoch, replacement=True)
        
        print(f"\nDataset-balanced sampling enabled: {num_samples_per_epoch:,} samples/epoch")
        print(f"  Target optimizer steps/epoch: {TARGET_OPT_STEPS_PER_EPOCH} ({'CPU-optimized' if str(DEVICE) == 'cpu' else 'GPU-optimized'})")
        print(f"  Effective batches/epoch: {num_samples_per_epoch // BATCH_SIZE}")
        print(f"  Contribution: CheXpert {p_chex*100:.0f}%, NIH {p_nih*100:.0f}%, Pneumonia {p_pneu*100:.0f}%")
        print(f"  (Prevents Pneumonia overfitting via capped repetition)")
        
        # Print dataset statistics
        print_dataset_stats("CheXpert Train", chexpert_train, SELECTED_LABELS)
        print_dataset_stats("NIH Train", nih_train, SELECTED_LABELS)
        print_dataset_stats("Pneumonia Train", pneumonia_train, SELECTED_LABELS)
        
        # CRITICAL: Verify NIH Effusion labels are being captured (must be >2000)
        eff_idx = LABEL_TO_IDX['Effusion']
        nih_train_eff = int(nih_train.labels[:, eff_idx].sum())
        nih_val_eff = int(nih_valid.labels[:, eff_idx].sum())
        print(f"\n{'='*70}")
        print("NIH EFFUSION LABEL VERIFICATION (CRITICAL)")
        print(f"{'='*70}")
        print(f"NIH Train Effusion positives: {nih_train_eff}")
        print(f"NIH Val   Effusion positives: {nih_val_eff}")
        if nih_train_eff < 2000:
            print("WARNING: NIH Train Effusion count too low! Expected >8000")
            print("This indicates label mapping may still be broken!")
        if nih_val_eff < 2000:
            print("WARNING: NIH Val Effusion count too low! Expected ~2800")
            print("This indicates label mapping may still be broken!")
        print(f"{'='*70}\n")
        
        # IMAGE PATH SANITY CHECK - catch CheXpert prefix/root_dir mismatches
        print(f"{'='*70}")
        print("IMAGE PATH SANITY CHECK")
        print(f"{'='*70}")
        quick_image_path_sanity(chexpert_train, "CheXpert Train", n=2000)
        quick_image_path_sanity(chexpert_valid, "CheXpert Val", n=2000)
        quick_image_path_sanity(nih_train, "NIH Train", n=2000)
        quick_image_path_sanity(nih_valid, "NIH Val", n=2000)
        print(f"{'='*70}\n")
        
        # OPTIMIZED DataLoaders
        print(f"\nDataLoader Config: num_workers={NUM_WORKERS}, prefetch_factor={PREFETCH_FACTOR}, batch_size={BATCH_SIZE}")
        
        train_loader = create_dataloader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            sampler=sampler,  # Use balanced sampler instead of shuffle
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS
        )
        
        # Use stratified validation loaders for consistent metrics (3K images per dataset)
        print("Creating stratified validation loaders (3K images each for stable metrics)...")
        chexpert_val_loader = create_stratified_sampled_loader(
            chexpert_valid, sample_size=3000, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
        )
        nih_val_loader = create_stratified_sampled_loader(
            nih_valid, sample_size=3000, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
        )
        pneumonia_val_loader = create_stratified_sampled_loader(
            pneumonia_valid, sample_size=min(3000, len(pneumonia_valid)), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
        )
        
        # Create model
        print("\nCreating model...")
        model, resume_info = create_model(num_classes=len(SELECTED_LABELS))
        model = model.to(DEVICE)
        
        # Apply channels-last memory format on CUDA for better bandwidth
        if str(DEVICE) == "cuda":
            model = model.to(memory_format=torch.channels_last)
            print("Applied channels-last memory format for CUDA optimization")
        
        # Print memory info
        print("\n")
        print_memory_info()
        
        # Optimizer and scheduler
        criterion = MaskedSmoothedFocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, smoothing=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        # CosineAnnealingWarmRestarts for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,        # Restart every 10 epochs
            T_mult=1,      # Keep same cycle length
            eta_min=1e-7   # Minimum learning rate
        )
        
        # PHASE 4: DO NOT resume optimizer/scheduler (fine-tune needs fresh LR + fresh moments)
        print("PHASE 4: Starting with FRESH optimizer/scheduler (not resuming state)")
        print(f"  Learning Rate: {LEARNING_RATE:.2e}")
        
        scaler = GradScaler(enabled=AMP_ENABLED) if HAS_AMP else None
        
        print(f"\nAMP: {'Enabled' if AMP_ENABLED else 'Disabled'}")
        print(f"Effective Batch Size: {BATCH_SIZE * ACCUMULATION_STEPS}")
        print(f"Validation Frequency: Every {VALIDATE_EVERY_N_EPOCHS} epochs")
        
        # Truncate log file to avoid appending across runs
        with open(LOG_FILE, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write(f"Configuration: BATCH_SIZE={BATCH_SIZE}, ACCUMULATION={ACCUMULATION_STEPS}, LR={LEARNING_RATE}\n")
            f.write(f"Dataset sizes - CheXpert: {len(chexpert_train)}, NIH: {len(nih_train)}, Pneumonia: {len(pneumonia_train)}\n\n")
        
        # Initialize detailed JSON log
        detailed_log = {
            "training_start": str(datetime.now()),
            "configuration": {
                "batch_size": BATCH_SIZE,
                "accumulation_steps": ACCUMULATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "finetune_epochs": FINETUNE_EPOCHS,
                "img_size": IMG_SIZE,
                "device": str(DEVICE),  # CRITICAL: JSON-safe serialization
                "amp_enabled": AMP_ENABLED,
                "clahe_enabled_train": enable_train_clahe,
                "dataset_probabilities": {"chexpert": p_chex, "nih": p_nih, "pneumonia": p_pneu}
            },
            "datasets": {
                "chexpert_train": len(chexpert_train),
                "nih_train": len(nih_train),
                "pneumonia_train": len(pneumonia_train),
                "total_concat_size": len(train_loader.dataset),
                "effective_samples_per_epoch": num_samples_per_epoch
            },
            "epochs": []
        }
        
        # Load existing log if resuming
        if os.path.exists(DETAILED_LOG_FILE) and RESUME_FROM_BEST:
            try:
                with open(DETAILED_LOG_FILE, 'r') as f:
                    detailed_log = json.load(f)
                print(f"Loaded existing detailed log with {len(detailed_log['epochs'])} epochs")
            except:
                print("Could not load existing log, creating new one")
        
        # Validate training setup
        if not validate_training_setup(model, train_loader, criterion, optimizer):
            print("ERROR: Exiting due to setup validation failure")
            sys.exit(1)
        
        # Training loop
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        best_auc = resume_info['best_auc']
        start_epoch = resume_info['start_epoch']
        end_epoch = start_epoch + FINETUNE_EPOCHS - 1
        patience_counter = 0
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"Fine-tuning Plan: Epoch {start_epoch} → {end_epoch} (total {FINETUNE_EPOCHS} epochs)")
        print(f"{'='*70}\n")
        
        for epoch in range(start_epoch, end_epoch + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, epoch, start_epoch, end_epoch)
            
            # Log epoch data
            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            
            epoch_data = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "learning_rate": float(current_lr),
                "epoch_time_seconds": float(epoch_time),
                "timestamp": str(datetime.now())
            }
            
            # Save emergency checkpoint before validation (prevents loss if validation crashes)
            emergency_checkpoint = os.path.join(CHECKPOINT_DIR, f"emergency_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'best_auc': best_auc
            }, emergency_checkpoint)
            
            # Rolling cleanup: keep only last 2 emergency checkpoints
            old_emergency = os.path.join(CHECKPOINT_DIR, f"emergency_epoch_{epoch-2}.pt")
            if os.path.exists(old_emergency):
                try:
                    os.remove(old_emergency)
                except:
                    pass
            
            # Validate every N epochs
            if epoch % VALIDATE_EVERY_N_EPOCHS == 0 or epoch == end_epoch:
                print(f"\n{'='*70}")
                print(f"EPOCH {epoch}/{end_epoch} - Train Loss: {train_loss:.4f}")
                print("="*70)
                
                print("\nNIH Validation:")
                nih_auc, nih_per_class = validate(model, nih_val_loader, "NIH")
                nih_eff_auc = nih_per_class.get('Effusion')
                nih_eff_auc = float(nih_eff_auc) if nih_eff_auc is not None else 0.0
                
                print("CheXpert Validation:")
                chexpert_auc, chexpert_per_class = validate(model, chexpert_val_loader, "CheXpert")
                
                print("Pneumonia Validation:")
                pneumonia_auc, pneumonia_per_class = validate(model, pneumonia_val_loader, "Pneumonia")
                
                mean_auc = (nih_auc + chexpert_auc + pneumonia_auc) / 3
                
                # CRITICAL FIX: Separate min AUC computation for core datasets vs Pneumonia
                # Pneumonia only has Consolidation labels, so including it in global min_auc
                # would allow false 85%+ achievement without truly getting 85% on all 5 diseases
                
                # Core datasets (NIH + CheXpert) must achieve good AUC on ALL 14 diseases
                min_auc_core = compute_min_auc([nih_per_class, chexpert_per_class])
                
                # Pneumonia dataset only has Pneumonia labels
                pneu_pneu = pneumonia_per_class.get("Pneumonia")
                pneu_pneu_auc = float(pneu_pneu) if pneu_pneu is not None else 0.0
                
                # Global min AUC for checkpoint criterion
                min_auc = min(min_auc_core, pneu_pneu_auc)
                
                # CRITICAL: Monitor key metrics with enhanced warnings
                print(f"\n{'='*70}")
                print("CRITICAL METRIC TRACKING - 14 DISEASES")
                print(f"{'='*70}")
                print(f"NIH Effusion AUC:              {nih_eff_auc:.4f}  <- Historically challenging")
                print(f"Pneumonia Binary AUC:          {pneu_pneu_auc:.4f}  <- Binary classification")
                print(f"Min AUC (Core 14 diseases):    {min_auc_core:.4f}  <- NIH+CheXpert all 14")
                print(f"Min AUC (Global):              {min_auc:.4f}  <- CHECKPOINT CRITERION")
                
                if epoch >= 10 and nih_eff_auc < 0.60:
                    print("\nWARNING: NIH Effusion AUC still low at epoch 10+")
                    print("   Expected >60% by epoch 10. Check if labels are being learned.")
                if epoch >= 10 and min_auc_core < 0.50:
                    print("\nWARNING: Core min AUC still low at epoch 10+")
                    print("   Expected >50% by epoch 10 for all 14 diseases (harder than 5).")
                if epoch >= 10 and pneu_pneu_auc < 0.75:
                    print("\nWARNING: Pneumonia AUC low at epoch 10+")
                    print("   Expected >75% by epoch 10 for binary classification.")
                print(f"{'='*70}\n")
                
                # CRITICAL FIX: Complete validation metrics with all components
                epoch_data["validation"] = {
                    "nih_auc": float(nih_auc),
                    "nih_effusion_auc": float(nih_eff_auc),  # Already 0.0 if None (no truthy check)
                    "chexpert_auc": float(chexpert_auc),
                    "pneumonia_auc": float(pneumonia_auc),
                    "pneumonia_pneumonia_auc": float(pneu_pneu_auc),  # Track separately
                    "mean_auc": float(mean_auc),
                    "min_auc_core": float(min_auc_core),  # Track core 14 diseases
                    "min_auc": float(min_auc)  # Global checkpoint criterion
                }
                epoch_data["is_best"] = False
                
                # Log results to text file
                with open(LOG_FILE, 'a') as f:
                    f.write(f"Epoch {epoch}/{end_epoch} | Train Loss: {train_loss:.4f} | "
                            f"NIH: {nih_auc:.4f} (Eff: {nih_eff_auc:.4f}) | CheXpert: {chexpert_auc:.4f} | "
                            f"Pneumonia: {pneumonia_auc:.4f} (Pneu: {pneu_pneu_auc:.4f}) | "
                            f"Mean: {mean_auc:.4f} | MinCore: {min_auc_core:.4f} | MinGlobal: {min_auc:.4f}\n")
                
                print(f"{'='*70}")
                print(f"SUMMARY - Epoch {epoch}/{end_epoch}")
                print(f"{'='*70}")
                print(f"Train Loss:                  {train_loss:.4f}")
                print(f"NIH AUC:                     {nih_auc:.4f}")
                print(f"CheXpert AUC:                {chexpert_auc:.4f}")
                print(f"Pneumonia AUC:               {pneumonia_auc:.4f}")
                print(f"Mean AUC:                    {mean_auc:.4f}")
                print(f"Min AUC (Core 14):           {min_auc_core:.4f}")
                print(f"Min AUC (Global):            {min_auc:.4f} <- CHECKPOINT (Best: {best_auc:.4f})")
                print(f"Learning Rate:               {optimizer.param_groups[0]['lr']:.2e}")
                print("="*70)
                
                # Save checkpoint based on global min AUC
                if min_auc > best_auc:
                    best_auc = min_auc
                    patience_counter = 0
                    epoch_data["is_best"] = True
                    
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_auc': best_auc,
                        'min_auc': min_auc,
                        'min_auc_core': min_auc_core,
                        'mean_auc': mean_auc,
                        'nih_auc': nih_auc,
                        'chexpert_auc': chexpert_auc,
                        'pneumonia_auc': pneumonia_auc,
                        'nih_effusion_auc': nih_eff_auc,
                        'pneumonia_pneumonia_auc': pneu_pneu_auc
                    }
                    
                    torch.save(checkpoint, BEST_MODEL_OUT)
                    print(f"\n🎉 NEW BEST MODEL SAVED! Min AUC: {min_auc:.4f}")
                    print(f"   Core (NIH+CheXpert 5 diseases): {min_auc_core:.4f}")
                    print(f"   Pneumonia (Pneumonia):          {pneu_pneu_auc:.4f}")
                    
                    # Clean up ALL emergency checkpoints after successful best model save
                    try:
                        emergency_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('emergency_epoch_')]
                        for file in emergency_files:
                            os.remove(os.path.join(CHECKPOINT_DIR, file))
                        if emergency_files:
                            print(f"Cleaned up {len(emergency_files)} emergency checkpoint(s)")
                    except Exception as e:
                        print(f"Warning: Emergency checkpoint cleanup failed: {e}")
                else:
                    patience_counter += 1
                    print(f"\nNo improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
                    
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print("\nEarly stopping triggered!")
                        break
            else:
                # For non-validation epochs, just mark that no validation occurred
                epoch_data["validation"] = None
                epoch_data["is_best"] = False
            
            # Save epoch data to detailed log after every epoch
            detailed_log["epochs"].append(epoch_data)
            with open(DETAILED_LOG_FILE, 'w') as f:
                json.dump(detailed_log, f, indent=2)
            
            # Scheduler already stepped per-batch inside train_one_epoch()
            # No need for per-epoch step with CosineAnnealingWarmRestarts
            
            # Periodic memory cleanup
            if epoch % 5 == 0 and str(DEVICE) == "cuda":
                torch.cuda.empty_cache()
                print("GPU cache cleared")
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            print(f"\nEpoch {epoch} time: {epoch_time/60:.1f} min | Total: {total_time/3600:.1f} hours\n")
        
        # Save final training metadata
        detailed_log["training_end"] = str(datetime.now())
        detailed_log["total_epochs_completed"] = len(detailed_log["epochs"])
        detailed_log["final_best_auc"] = float(best_auc)
        with open(DETAILED_LOG_FILE, 'w') as f:
            json.dump(detailed_log, f, indent=2)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Detailed logs saved to: {DETAILED_LOG_FILE}")
        
        # Clean up temporary files
        cleanup_temp_files()
        
        print(f"\nBest Min AUC: {best_auc:.4f}  # Checkpoint criterion: worst-case class")
        print(f"Total Time: {(time.time() - start_time)/3600:.2f} hours")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving checkpoint...")
        if 'epoch' in locals() and 'model' in locals():
            emergency_save_path = os.path.join(CHECKPOINT_DIR, 'interrupted_checkpoint.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if 'optimizer' in locals() else None,
                'scheduler_state_dict': scheduler.state_dict() if 'scheduler' in locals() else None,
                'best_auc': best_auc if 'best_auc' in locals() else 0.0
            }, emergency_save_path)
            print(f"Emergency checkpoint saved to: {emergency_save_path}")
        print("Training interrupted successfully.")
        
    except Exception as e:
        print(f"\nUnexpected error occurred: {str(e)}")
        print("Check logs and system resources.")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main()
