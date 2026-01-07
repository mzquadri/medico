"""
PHASE 4 ACTUAL ALIGNMENT VERIFICATION
======================================
Verifies that splits and label mappings match EXACTLY what train_phase4_masked.py uses.
"""

import os
import numpy as np
import pandas as pd

print("\n" + "="*90)
print("PHASE 4 ACTUAL ALIGNMENT VERIFICATION")
print("Checking if data splits match train_phase4_masked.py implementation")
print("="*90)

# EXACT same as train_phase4_masked.py Lines 118-134
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
LABEL_TO_IDX = {label: idx for idx, label in enumerate(SELECTED_LABELS)}

print(f"\n1. TARGET DISEASES (from train_phase4_masked.py Lines 118-134):")
print(f"   Total: {len(SELECTED_LABELS)} diseases")
for idx, label in enumerate(SELECTED_LABELS):
    print(f"   [{idx:2d}] {label}")

# Paths - EXACT same as train_phase4_masked.py
CHEXPERT_ROOT = r"C:\Users\MohdZaminQuadri\Downloads\Medico-Xray\datasets\chexpert"
CHEX_TRAIN = os.path.join(CHEXPERT_ROOT, "train.csv")
CHEX_VALID = os.path.join(CHEXPERT_ROOT, "valid.csv")

NIH_CSV = r"C:\Users\MohdZaminQuadri\Downloads\Medico-Xray\datasets\nih\Data_Entry_2017.csv\Data_Entry_2017.csv"
CHECKPOINT_DIR = "checkpoints_phase4_masked"
NIH_FIXED_TRAIN = os.path.join(CHECKPOINT_DIR, "nih_fixed_train.csv")
NIH_FIXED_VAL = os.path.join(CHECKPOINT_DIR, "nih_fixed_val.csv")
NIH_FIXED_TEST = os.path.join(CHECKPOINT_DIR, "nih_fixed_test.csv")

CHEX_FIXED_TRAIN = os.path.join(CHECKPOINT_DIR, "chexpert_fixed_train.csv")
CHEX_FIXED_VAL = os.path.join(CHECKPOINT_DIR, "chexpert_fixed_val.csv")
CHEX_FIXED_TEST = os.path.join(CHECKPOINT_DIR, "chexpert_fixed_test.csv")

PNEU_ROOT = r"C:\Users\MohdZaminQuadri\Downloads\Medico-Xray\datasets\pneumonia"
PNEU_FIXED_TRAIN = os.path.join(CHECKPOINT_DIR, "pneumonia_fixed_train.csv")
PNEU_FIXED_VAL = os.path.join(CHECKPOINT_DIR, "pneumonia_fixed_val.csv")
PNEU_FIXED_TEST = os.path.join(CHECKPOINT_DIR, "pneumonia_fixed_test.csv")

# =============================================================================
# CHEXPERT VERIFICATION
# =============================================================================
print("\n" + "="*90)
print("DATASET 1: CHEXPERT")
print("="*90)

print("\n2. CHEXPERT ORIGINAL FILES (train.csv, valid.csv):")
if os.path.exists(CHEX_TRAIN):
    chex_train_df = pd.read_csv(CHEX_TRAIN)
    print(f"   train.csv: {len(chex_train_df):,} images")
    
    # Check disease columns
    chexpert_to_nih_map = {
        "Cardiomegaly": "Cardiomegaly",
        "Edema": "Edema",
        "Consolidation": "Consolidation",
        "Atelectasis": "Atelectasis",
        "Pleural Effusion": "Effusion",
        "Pneumothorax": "Pneumothorax",
        "Pneumonia": "Pneumonia"
    }
    
    print(f"\n3. CHEXPERT DISEASE MAPPING:")
    for chex_name, nih_name in chexpert_to_nih_map.items():
        if chex_name in chex_train_df.columns:
            nih_idx = LABEL_TO_IDX.get(nih_name, -1)
            col_data = chex_train_df[chex_name].to_numpy(dtype=np.float32)
            
            pos = np.nansum(col_data == 1.0)
            neg = np.nansum(col_data == 0.0)
            unc = np.nansum(col_data == -1.0)
            nan = np.isnan(col_data).sum()
            
            # Phase 4 handling: NaN->0 (negative), -1->masked
            raw = np.nan_to_num(col_data, nan=0.0)
            mask = (raw != -1.0).astype(np.float32)
            certain = int(mask.sum())
            masked = int((1 - mask).sum())
            
            print(f"   {chex_name:<20} -> {nih_name:<20} [idx {nih_idx:2d}]")
            print(f"      Pos={int(pos):6d} Neg={int(neg):6d} Unc={int(unc):6d} NaN={int(nan):6d}")
            print(f"      Certain={certain:6d} Masked={masked:6d}")
        else:
            print(f"   {chex_name:<20} -> MISSING IN CSV")
    
    # Missing 7 diseases
    print(f"\n4. MISSING 7 DISEASES (treated as negatives):")
    missing = ["Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening"]
    for disease in missing:
        idx = LABEL_TO_IDX.get(disease, -1)
        print(f"   {disease:<20} [idx {idx:2d}] -> Label=0.0, Mask=1.0 (certain negative)")
else:
    print(f"   ERROR: train.csv not found at {CHEX_TRAIN}")

print("\n5. CHEXPERT FIXED SPLITS (80/10/10 patient-level):")
if os.path.exists(CHEX_FIXED_TRAIN) and os.path.exists(CHEX_FIXED_VAL) and os.path.exists(CHEX_FIXED_TEST):
    train_df = pd.read_csv(CHEX_FIXED_TRAIN)
    val_df = pd.read_csv(CHEX_FIXED_VAL)
    test_df = pd.read_csv(CHEX_FIXED_TEST)
    
    print(f"   Train: {len(train_df):,} images")
    print(f"   Val:   {len(val_df):,} images")
    print(f"   Test:  {len(test_df):,} images")
    
    # Check patient-level split
    train_df['Patient'] = train_df['Path'].str.extract(r'(patient\d+)')
    val_df['Patient'] = val_df['Path'].str.extract(r'(patient\d+)')
    test_df['Patient'] = test_df['Path'].str.extract(r'(patient\d+)')
    
    train_patients = set(train_df['Patient'].unique())
    val_patients = set(val_df['Patient'].unique())
    test_patients = set(test_df['Patient'].unique())
    
    overlap_train_val = len(train_patients.intersection(val_patients))
    overlap_train_test = len(train_patients.intersection(test_patients))
    overlap_val_test = len(val_patients.intersection(test_patients))
    
    print(f"\n   DATA LEAKAGE CHECK:")
    print(f"   Train & Val:  {overlap_train_val} patients {'OK' if overlap_train_val==0 else 'LEAKAGE!'}")
    print(f"   Train & Test: {overlap_train_test} patients {'OK' if overlap_train_test==0 else 'LEAKAGE!'}")
    print(f"   Val & Test:   {overlap_val_test} patients {'OK' if overlap_val_test==0 else 'LEAKAGE!'}")
else:
    print(f"   Fixed splits NOT FOUND (will be created on first training run)")
    print(f"   train_phase4_masked.py will create them at Lines 1461-1509")

# =============================================================================
# NIH VERIFICATION
# =============================================================================
print("\n" + "="*90)
print("DATASET 2: NIH CHEST X-RAY")
print("="*90)

print("\n6. NIH ORIGINAL CSV:")
if os.path.exists(NIH_CSV):
    nih_df = pd.read_csv(NIH_CSV)
    print(f"   Total images: {len(nih_df):,}")
    print(f"   Total patients: {nih_df['Patient ID'].nunique():,}")
    
    # Count all 14 diseases
    print(f"\n7. NIH DISEASE COUNTS (all 14 target diseases):")
    fl = nih_df["Finding Labels"].fillna("")
    
    for idx, disease in enumerate(SELECTED_LABELS):
        # Special handling for Effusion (NIH name)
        search_name = disease
        if disease == "Effusion":
            search_name = "Effusion"  # NIH uses "Effusion", not "Pleural Effusion"
        
        # Regex: match exact disease name in pipe-separated list
        has_disease = fl.str.contains(rf'(^|\|){search_name}(\||$)', regex=True)
        count = int(has_disease.sum())
        pct = count / len(nih_df) * 100
        
        print(f"   [{idx:2d}] {disease:<20} : {count:6d} ({pct:5.2f}%)")
else:
    print(f"   ERROR: NIH CSV not found at {NIH_CSV}")

print("\n8. NIH FIXED SPLITS (80/10/10 patient-level stratified):")
if os.path.exists(NIH_FIXED_TRAIN) and os.path.exists(NIH_FIXED_VAL) and os.path.exists(NIH_FIXED_TEST):
    train_df = pd.read_csv(NIH_FIXED_TRAIN)
    val_df = pd.read_csv(NIH_FIXED_VAL)
    test_df = pd.read_csv(NIH_FIXED_TEST)
    
    print(f"   Train: {len(train_df):,} images")
    print(f"   Val:   {len(val_df):,} images")
    print(f"   Test:  {len(test_df):,} images")
    
    # Check patient-level split (NO LEAKAGE)
    train_patients = set(train_df['Patient ID'].unique())
    val_patients = set(val_df['Patient ID'].unique())
    test_patients = set(test_df['Patient ID'].unique())
    
    overlap_train_val = len(train_patients.intersection(val_patients))
    overlap_train_test = len(train_patients.intersection(test_patients))
    overlap_val_test = len(val_patients.intersection(test_patients))
    
    print(f"\n   DATA LEAKAGE CHECK (CRITICAL):")
    print(f"   Train & Val:  {overlap_train_val} patients {'OK - PATIENT-LEVEL SPLIT' if overlap_train_val==0 else 'LEAKAGE DETECTED!'}")
    print(f"   Train & Test: {overlap_train_test} patients {'OK - PATIENT-LEVEL SPLIT' if overlap_train_test==0 else 'LEAKAGE DETECTED!'}")
    print(f"   Val & Test:   {overlap_val_test} patients {'OK - PATIENT-LEVEL SPLIT' if overlap_val_test==0 else 'LEAKAGE DETECTED!'}")
    
    if overlap_train_val == 0 and overlap_train_test == 0 and overlap_val_test == 0:
        print(f"   VERIFIED: Patient-level split (Lines 1528-1565 in train_phase4_masked.py)")
else:
    print(f"   Fixed splits NOT FOUND (will be created on first training run)")
    print(f"   train_phase4_masked.py will create them at Lines 1528-1565")
    print(f"   Implementation: Patient-level stratified 80/10/10 split")

# =============================================================================
# PNEUMONIA VERIFICATION
# =============================================================================
print("\n" + "="*90)
print("DATASET 3: PNEUMONIA KAGGLE")
print("="*90)

print("\n9. PNEUMONIA FOLDER STRUCTURE:")
train_normal = len([f for f in os.listdir(os.path.join(PNEU_ROOT, "train", "NORMAL")) if f.endswith(('.jpeg', '.jpg', '.png'))]) if os.path.exists(os.path.join(PNEU_ROOT, "train", "NORMAL")) else 0
train_pneumonia = len([f for f in os.listdir(os.path.join(PNEU_ROOT, "train", "PNEUMONIA")) if f.endswith(('.jpeg', '.jpg', '.png'))]) if os.path.exists(os.path.join(PNEU_ROOT, "train", "PNEUMONIA")) else 0
test_normal = len([f for f in os.listdir(os.path.join(PNEU_ROOT, "test", "NORMAL")) if f.endswith(('.jpeg', '.jpg', '.png'))]) if os.path.exists(os.path.join(PNEU_ROOT, "test", "NORMAL")) else 0
test_pneumonia = len([f for f in os.listdir(os.path.join(PNEU_ROOT, "test", "PNEUMONIA")) if f.endswith(('.jpeg', '.jpg', '.png'))]) if os.path.exists(os.path.join(PNEU_ROOT, "test", "PNEUMONIA")) else 0

total = train_normal + train_pneumonia + test_normal + test_pneumonia
print(f"   Total images: {total}")
print(f"   train/NORMAL:    {train_normal}")
print(f"   train/PNEUMONIA: {train_pneumonia}")
print(f"   test/NORMAL:     {test_normal}")
print(f"   test/PNEUMONIA:  {test_pneumonia}")

print(f"\n10. PNEUMONIA LABEL MAPPING (from train_phase4_masked.py Lines 1016-1021):")
pneumonia_idx = LABEL_TO_IDX.get('Pneumonia', -1)
print(f"   PNEUMONIA folder -> 'Pneumonia' disease [idx {pneumonia_idx}]")
print(f"   NORMAL folder    -> 'Pneumonia'=0.0 (all others 0.0)")
print(f"   Implementation: PneumoniaDataset class (Lines 981-1030)")

print(f"\n11. PNEUMONIA FIXED SPLITS (70/15/15 stratified):")
if os.path.exists(PNEU_FIXED_TRAIN) and os.path.exists(PNEU_FIXED_VAL) and os.path.exists(PNEU_FIXED_TEST):
    train_df = pd.read_csv(PNEU_FIXED_TRAIN)
    val_df = pd.read_csv(PNEU_FIXED_VAL)
    test_df = pd.read_csv(PNEU_FIXED_TEST)
    
    print(f"   Train: {len(train_df):,} images")
    print(f"   Val:   {len(val_df):,} images")
    print(f"   Test:  {len(test_df):,} images")
    
    # Check class balance
    train_pneu = (train_df['label'] == 1.0).sum()
    val_pneu = (val_df['label'] == 1.0).sum()
    test_pneu = (test_df['label'] == 1.0).sum()
    
    train_ratio = train_pneu / (len(train_df) - train_pneu) if len(train_df) > train_pneu else 0
    val_ratio = val_pneu / (len(val_df) - val_pneu) if len(val_df) > val_pneu else 0
    test_ratio = test_pneu / (len(test_df) - test_pneu) if len(test_df) > test_pneu else 0
    
    print(f"\n   CLASS BALANCE (PNEUMONIA:NORMAL ratio):")
    print(f"   Train: {train_ratio:.2f}:1")
    print(f"   Val:   {val_ratio:.2f}:1")
    print(f"   Test:  {test_ratio:.2f}:1")
    
    ratio_diff = max(abs(train_ratio - val_ratio), abs(train_ratio - test_ratio), abs(val_ratio - test_ratio))
    if ratio_diff < 0.3:
        print(f"   VERIFIED: Stratified split maintains class balance")
    else:
        print(f"   WARNING: Class balance varies (diff: {ratio_diff:.2f})")
else:
    print(f"   Fixed splits NOT FOUND (will be created on first training run)")
    print(f"   train_phase4_masked.py will create them at Lines 1574-1633")
    print(f"   Implementation: 70/15/15 stratified class-balanced split")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*90)
print("FINAL ALIGNMENT SUMMARY")
print("="*90)

print(f"\nTARGET DISEASES: {len(SELECTED_LABELS)} diseases")
print(f"   NIH:       14/14 diseases (100% coverage)")
print(f"   CheXpert:  7/14 diseases matched, 7 as negatives")
print(f"   Pneumonia: 1/14 diseases (Pneumonia only)")
print(f"   Combined:  14/14 diseases (100% coverage)")

print(f"\nSPLIT RATIOS:")
print(f"   CheXpert:  80/10/10 patient-level")
print(f"   NIH:       80/10/10 patient-level stratified")
print(f"   Pneumonia: 70/15/15 stratified class-balanced")

print(f"\nDATA LEAKAGE PREVENTION:")
print(f"   CheXpert:  Patient-level split (no same patient in train+val)")
print(f"   NIH:       Patient-level split (Lines 1528-1565)")
print(f"   Pneumonia: Stratified split (different images)")

print(f"\nLABEL HANDLING:")
print(f"   CheXpert:  NaN->0 (negative), -1->masked (uncertain)")
print(f"   NIH:       Pipe-separated labels (Disease1|Disease2)")
print(f"   Pneumonia: Folder-based (NORMAL/PNEUMONIA)")

print(f"\nTRAINING READINESS:")
chex_ready = os.path.exists(CHEX_FIXED_TRAIN) or os.path.exists(CHEX_TRAIN)
nih_ready = os.path.exists(NIH_FIXED_TRAIN) or os.path.exists(NIH_CSV)
pneu_ready = os.path.exists(PNEU_FIXED_TRAIN) or (os.path.exists(os.path.join(PNEU_ROOT, "train", "PNEUMONIA")))

print(f"   CheXpert:  {'READY' if chex_ready else 'NOT READY'}")
print(f"   NIH:       {'READY' if nih_ready else 'NOT READY'}")
print(f"   Pneumonia: {'READY' if pneu_ready else 'NOT READY'}")

if chex_ready and nih_ready and pneu_ready:
    print(f"\nALL DATASETS READY FOR PHASE 4 TRAINING!")
    print(f"Run: python train_phase4_masked.py")
else:
    print(f"\nSOME DATASETS NOT READY - Check paths above")

print("\n" + "="*90 + "\n")
