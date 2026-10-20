"""
Chest X-Ray Disease Detection - Prediction Script

Command-line tool for analyzing chest X-ray images and detecting
common thoracic diseases using a pre-trained deep learning model.

Usage:
    python predict.py <image_path>

Example:
    python predict.py chest_xray.png
"""

import sys

import numpy as np
import skimage.io
import skimage.transform
import torch
from model_handler import XRayModel


def preprocess_image(image_path):
    """
    Load and preprocess a chest X-ray image for model inference.

    The preprocessing steps include:
    1. Loading the image from disk
    2. Converting to grayscale if necessary
    3. Normalizing pixel values to [-1024, 1024] range (HU units)
    4. Resizing to 224x224 pixels
    5. Converting to PyTorch tensor

    Args:
        image_path: Path to the X-ray image file

    Returns:
        PyTorch tensor of shape (1, 1, 224, 224)
    """
    img = skimage.io.imread(image_path)
    img = img.astype(np.float32)

    # Convert RGB to grayscale
    if len(img.shape) == 3:
        img = img.mean(axis=2)

    # Normalize to Hounsfield units range
    img = (img / 255.0) * 2048 - 1024

    # Resize to model input size
    img = skimage.transform.resize(img, (224, 224), preserve_range=True)

    # Create tensor with batch and channel dimensions
    img_tensor = torch.from_numpy(img[np.newaxis, ...]).float().unsqueeze(0)

    return img_tensor


def analyze_xray(image_path):
    """
    Analyze a chest X-ray image for disease detection.

    Args:
        image_path: Path to the X-ray image file

    Returns:
        Dictionary containing prediction results for each disease
    """
    print("Loading model...")
    model = XRayModel()
    model.load_model()

    print(f"Processing image: {image_path}")
    img_tensor = preprocess_image(image_path)

    results = model.predict_diseases(img_tensor)

    print("\n" + "-" * 50)
    print("ANALYSIS RESULTS")
    print("-" * 50)

    detected = []
    for disease, data in results.items():
        prob = data["probability"]
        is_detected = data["detected"]

        status = "[DETECTED]" if is_detected else "[negative]"
        print(f"{disease}: {prob:.1%} {status}")

        if is_detected:
            detected.append(disease)

    print("-" * 50)

    if detected:
        print(f"\nFindings: {', '.join(detected)}")
    else:
        print("\nNo abnormalities detected")

    return results


def main():
    if len(sys.argv) < 2:
        print("Chest X-Ray Disease Detection")
        print("-" * 30)
        print("Usage: python predict.py <image_path>")
        print("\nSupported formats: PNG, JPEG, DICOM")
        sys.exit(1)

    image_path = sys.argv[1]
    analyze_xray(image_path)


if __name__ == "__main__":
    main()
