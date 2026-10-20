"""
Chest X-Ray Disease Detection - Image Processing Utilities

This module provides functions for loading and preprocessing
chest X-ray images for model inference.
"""

import numpy as np
import torch
from PIL import Image
from skimage.transform import resize


def load_image(image_path):
    """
    Load an X-ray image from file.

    Args:
        image_path: Path to the image file

    Returns:
        Grayscale image as numpy array
    """
    img = Image.open(image_path)

    if img.mode != 'L':
        img = img.convert('L')

    return np.array(img)


def normalize_image(img_array):
    """
    Normalize image to Hounsfield units range [-1024, 1024].

    This normalization is required by the TorchXRayVision models
    which were trained on images in this range.

    Args:
        img_array: Input image as numpy array (0-255 range)

    Returns:
        Normalized image in [-1024, 1024] range
    """
    img_float = img_array.astype(np.float32)
    return (img_float / 255.0) * 2048 - 1024


def resize_image(img_array, target_size=224):
    """
    Resize image to the target dimensions.

    Args:
        img_array: Input image as numpy array
        target_size: Output size (default 224 for DenseNet)

    Returns:
        Resized image
    """
    return resize(img_array, (target_size, target_size),
                  anti_aliasing=True, preserve_range=True)


def to_tensor(img_array):
    """
    Convert numpy array to PyTorch tensor with correct dimensions.

    Args:
        img_array: 2D numpy array (height, width)

    Returns:
        Tensor of shape (1, 1, height, width)
    """
    tensor = torch.from_numpy(img_array).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor


def process_xray(image_path, size=224):
    """
    Complete preprocessing pipeline for chest X-ray images.

    Args:
        image_path: Path to the X-ray image
        size: Target size for resizing (default 224)

    Returns:
        Preprocessed tensor ready for model inference
    """
    img = load_image(image_path)
    img = normalize_image(img)
    img = resize_image(img, size)
    tensor = to_tensor(img)

    return tensor
