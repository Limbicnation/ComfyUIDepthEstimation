"""
ComfyUI Depth Estimation Node - Verified and Enhanced Version
A custom node for depth map estimation using transformer models.
"""

import os
import numpy as np
import torch
from transformers import pipeline
from PIL import Image, ImageFilter, ImageOps
import folder_paths
from comfy.model_management import get_torch_device

def ensure_odd(value):
    """Ensure the value is an odd integer."""
    value = int(value)
    return value if value % 2 == 1 else value + 1

def convert_path(path):
    """Convert path for compatibility between Windows and WSL."""
    if os.name == 'nt':
        return path.replace('\\', '/')
    return path

def gamma_correction(img, gamma=1.0):
    """Apply gamma correction to the image."""
    if not isinstance(img, (Image.Image, np.ndarray)):
        raise TypeError("Input must be PIL Image or numpy array")
    
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return Image.fromarray(np.array(img).astype(np.uint8)).point(lambda i: table[i])

def auto_gamma_correction(image):
    """Automatically adjust gamma correction for the image."""
    image_array = np.array(image).astype(np.float32) / 255.0
    mean_luminance = np.mean(image_array)
    if mean_luminance <= 0:
        return image
    gamma = np.log(0.5) / np.log(mean_luminance)
    return gamma_correction(image, gamma=gamma)

def auto_contrast(image):
    """Apply automatic contrast adjustment to the image."""
    return ImageOps.autocontrast(image)

class DepthEstimationNode:
    def __init__(self):
        self.device = get_torch_device()
        self.depth_estimator = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0}),
                "median_size": ("INT", {"default": 5, "min": 3, "max": 11, "step": 2}),
                "apply_auto_contrast": ("BOOLEAN", {"default": True}),
                "apply_gamma": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_depth"
    CATEGORY = "image/processing"

    def ensure_model_loaded(self):
        """Ensure the depth estimation model is loaded."""
        if self.depth_estimator is None:
            try:
                self.depth_estimator = pipeline("depth-estimation", device=self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load depth estimation model: {str(e)}")

    def estimate_depth(self, image, blur_radius=2.0, median_size=5, 
                      apply_auto_contrast=True, apply_gamma=True):
        """
        Estimate depth from input image with optional post-processing.
        
        Args:
            image (torch.Tensor): Input image tensor (B,H,W,C)
            blur_radius (float): Gaussian blur radius
            median_size (int): Median filter kernel size
            apply_auto_contrast (bool): Whether to apply automatic contrast
            apply_gamma (bool): Whether to apply gamma correction
            
        Returns:
            tuple(torch.Tensor): Processed depth map tensor (B,H,W)
        """
        self.ensure_model_loaded()
        
        # Input validation
        if not isinstance(image, np.ndarray) or image.ndim != 4:
            raise ValueError("Input image must be 4D numpy array (B,H,W,C)")
        
        # Convert image to PIL
        image_pil = Image.fromarray((image[0] * 255).astype(np.uint8))
        
        try:
            # Generate depth map
            depth_map = self.depth_estimator(image_pil)["depth"]
            
            # Post-processing pipeline
            median_size = ensure_odd(median_size)  # Ensure odd kernel size
            
            if blur_radius > 0:
                depth_map = depth_map.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            depth_map = depth_map.filter(ImageFilter.MedianFilter(size=median_size))
            
            if apply_auto_contrast:
                depth_map = auto_contrast(depth_map)
            
            if apply_gamma:
                depth_map = auto_gamma_correction(depth_map)
            
            # Convert to tensor format
            depth_tensor = np.array(depth_map).astype(np.float32) / 255.0
            depth_tensor = depth_tensor[None, ...]  # Add batch dimension
            
            return (depth_tensor,)
            
        except Exception as e:
            raise RuntimeError(f"Depth estimation failed: {str(e)}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "DepthEstimationNode": DepthEstimationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthEstimationNode": "Depth Estimation"
}