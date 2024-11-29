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
    if os.name == 'nt':  # If running on Windows
        return path.replace('\\', '/')
    return path

def gamma_correction(img, gamma=1.0):
    """Apply gamma correction to the image."""
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return Image.fromarray(np.array(img).astype(np.uint8)).point(lambda i: table[i])

def auto_gamma_correction(image):
    """Automatically adjust gamma correction for the image."""
    image_array = np.array(image).astype(np.float32) / 255.0
    mean_luminance = np.mean(image_array)
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
                "median_size": ("INT", {"default": 5, "min": 3, "max": 11, "step": 2})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_depth"
    CATEGORY = "image/depth"

    def ensure_model_loaded(self):
        if self.depth_estimator is None:
            self.depth_estimator = pipeline("depth-estimation", device=self.device)

    def estimate_depth(self, image, blur_radius=2.0, median_size=5):
        self.ensure_model_loaded()
        
        # Convert image to PIL
        image_pil = Image.fromarray((image[0] * 255).astype(np.uint8))
        
        # Process image
        depth_map = self.depth_estimator(image_pil)["depth"]
        
        # Post-processing
        depth_map = depth_map.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        depth_map = depth_map.filter(ImageFilter.MedianFilter(size=median_size))
        
        # Convert back to tensor format
        depth_tensor = np.array(depth_map).astype(np.float32) / 255.0
        depth_tensor = depth_tensor[None, ...]
        
        return (depth_tensor,)

NODE_CLASS_MAPPINGS = {
    "DepthEstimationNode": DepthEstimationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthEstimationNode": "Depth Estimation"
}
