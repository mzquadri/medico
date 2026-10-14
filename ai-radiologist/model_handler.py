"""
Chest X-Ray Disease Detection - Model Handler

This module handles loading and inference using a pre-trained DenseNet model
for detecting common thoracic diseases from chest radiographs.

The model is trained on multiple large-scale datasets including NIH ChestX-ray14,
CheXpert, MIMIC-CXR, and PadChest, covering approximately 800,000 images.
"""

import os
import torch
import torchxrayvision as xrv
from typing import Dict, List
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "densenet121-res224-all.pt")

# All pathologies supported by the model
PATHOLOGIES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
    'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'
]

# Primary diseases for screening (based on CheXpert competition)
TARGET_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion"
]

# Detection thresholds calibrated on NIH dataset (5606 images)
# These values balance sensitivity and specificity for clinical use
THRESHOLDS = {
    "Atelectasis": 0.58,
    "Cardiomegaly": 0.55,
    "Consolidation": 0.60,
    "Edema": 0.52,
    "Effusion": 0.61,
}


class XRayModel:
    """
    Wrapper class for the chest X-ray disease detection model.
    
    Uses DenseNet-121 architecture pre-trained on multiple chest X-ray datasets.
    Supports detection of 18 different thoracic conditions.
    """
    
    def __init__(self, model_name="densenet121-res224-all"):
        self.model = None
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pathologies = PATHOLOGIES
        
    def load_model(self):
        """Load the pre-trained model weights."""
        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}")
        
        if os.path.exists(MODEL_PATH):
            print(f"Loading from: {MODEL_PATH}")
            self.model = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        else:
            print("Local weights not found, downloading from torchxrayvision...")
            self.model = xrv.models.DenseNet(weights=self.model_name)
            
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully")
        
    def predict(self, img_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Run inference on a preprocessed image tensor.
        
        Args:
            img_tensor: Preprocessed image tensor of shape (1, 1, 224, 224)
            
        Returns:
            Dictionary mapping pathology names to probability scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
        return dict(zip(self.pathologies, probs))
    
    def predict_diseases(self, img_tensor: torch.Tensor) -> Dict[str, dict]:
        """
        Run inference and return results for the 5 target diseases.
        
        Args:
            img_tensor: Preprocessed image tensor
            
        Returns:
            Dictionary with disease predictions including probability,
            threshold, and detection status
        """
        predictions = self.predict(img_tensor)
        
        results = {}
        for disease in TARGET_DISEASES:
            prob = None
            for key in predictions:
                if disease.lower() == key.lower():
                    prob = float(predictions[key])
                    break
            
            if prob is not None:
                threshold = THRESHOLDS.get(disease, 0.5)
                results[disease] = {
                    "probability": prob,
                    "threshold": threshold,
                    "detected": prob >= threshold
                }
        
        return results


def filter_detected(predictions: Dict[str, dict]) -> List[Dict]:
    """
    Extract only the detected diseases from prediction results.
    
    Args:
        predictions: Output from predict_diseases()
        
    Returns:
        List of detected diseases sorted by probability (highest first)
    """
    detected = []
    
    for disease, data in predictions.items():
        if data["detected"]:
            detected.append({
                "disease": disease,
                "probability": data["probability"],
                "threshold": data["threshold"]
            })
    
    detected.sort(key=lambda x: x["probability"], reverse=True)
    return detected


def filter_negative(predictions: Dict[str, dict]) -> List[Dict]:
    """
    Extract diseases that were not detected from prediction results.
    
    Args:
        predictions: Output from predict_diseases()
        
    Returns:
        List of negative diseases sorted by probability (lowest first)
    """
    negative = []
    
    for disease, data in predictions.items():
        if not data["detected"]:
            negative.append({
                "disease": disease,
                "probability": data["probability"],
                "threshold": data["threshold"]
            })
    
    negative.sort(key=lambda x: x["probability"])
    return negative
