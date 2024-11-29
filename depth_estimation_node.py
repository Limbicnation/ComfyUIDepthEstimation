import os
import numpy as np
import torch
from transformers import pipeline
from PIL import Image, ImageFilter, ImageOps
import folder_paths
from comfy.model_management import get_torch_device

DEPTH_MODELS = {
    "Depth-Anything-Small": "LiheYoung/depth-anything-small",
    "Depth-Anything-Base": "LiheYoung/depth-anything-base",
    "Depth-Anything-Large": "LiheYoung/depth-anything-large",
    "Depth-Anything-V2-Small": "LiheYoung/depth-anything-small-hf",
    "Depth-Anything-V2-Base": "LiheYoung/depth-anything-base-hf",
}

class DepthEstimationNode:
    MEDIAN_SIZES = ["3", "5", "7", "9", "11"]  # Valid median sizes

    def __init__(self):
        self.device = get_torch_device()
        self.depth_estimator = None
        self.current_model = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (list(DEPTH_MODELS.keys()),),
                "blur_radius": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "median_size": (cls.MEDIAN_SIZES, {"default": "5"}),  # Added default value
                "apply_auto_contrast": ("BOOLEAN", {"default": True}),
                "apply_gamma": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_depth"
    CATEGORY = "image/depth"

    def ensure_model_loaded(self, model_name):
        """Ensure the depth estimation model is loaded."""
        model_path = DEPTH_MODELS[model_name]
        if self.depth_estimator is None or self.current_model != model_path:
            try:
                self.depth_estimator = pipeline(
                    "depth-estimation",
                    model=model_path,
                    device=self.device
                )
                self.current_model = model_path
            except Exception as e:
                raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    def estimate_depth(self, image, model_name, blur_radius=2.0, median_size="5", 
                      apply_auto_contrast=True, apply_gamma=True):
        """
        Estimate depth from input image with optional post-processing.
        """
        try:
            # Ensure median_size is a string
            median_size = str(median_size)
            
            # Validate median_size
            if median_size not in self.MEDIAN_SIZES:
                raise ValueError(f"Invalid median_size. Must be one of {self.MEDIAN_SIZES}")
            
            median_size_int = int(median_size)
            self.ensure_model_loaded(model_name)
            
            # Convert tensor to numpy if needed
            if torch.is_tensor(image):
                image = image.cpu().numpy()
                
            # Ensure image is in range [0, 1]
            if image.max() > 1.0:
                image = image / 255.0
                
            # Convert to RGB if necessary
            if image.shape[-1] == 4:  # RGBA to RGB
                image = image[..., :3]
                
            # Convert to PIL Image for processing
            pil_image = Image.fromarray((image[0] * 255).astype(np.uint8))
            
            try:
                # Generate depth map
                depth_map = self.depth_estimator(pil_image)["predicted_depth"]
                
                # Convert depth map to PIL Image if it's not already
                if not isinstance(depth_map, Image.Image):
                    # Normalize depth values to 0-255 range
                    depth_map = ((depth_map - depth_map.min()) * (255 / (depth_map.max() - depth_map.min()))).astype(np.uint8)
                    depth_map = Image.fromarray(depth_map)
                
                # Post-processing pipeline
                if blur_radius > 0:
                    depth_map = depth_map.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                
                if median_size_int > 0:
                    depth_map = depth_map.filter(ImageFilter.MedianFilter(size=median_size_int))
                
                if apply_auto_contrast:
                    depth_map = ImageOps.autocontrast(depth_map)
                
                if apply_gamma:
                    depth_array = np.array(depth_map).astype(np.float32) / 255.0
                    mean_luminance = np.mean(depth_array)
                    if mean_luminance > 0:
                        gamma = np.log(0.5) / np.log(mean_luminance)
                        depth_map = self.gamma_correction(depth_map, gamma)
                
                # Convert to numpy array and normalize
                depth_array = np.array(depth_map).astype(np.float32) / 255.0
                
                # Add batch and channel dimensions to match ComfyUI format (B,H,W,C)
                depth_tensor = depth_array[None, ..., None]
                
                return (depth_tensor,)
                
            except Exception as e:
                raise RuntimeError(f"Depth estimation failed: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"Depth estimation failed: {str(e)}")

    def gamma_correction(self, img, gamma=1.0):
        """Apply gamma correction to the image."""
        inv_gamma = 1.0 / gamma
        table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        return Image.fromarray(np.array(img).astype(np.uint8)).point(lambda i: table[i])

# Node registration
NODE_CLASS_MAPPINGS = {
    "DepthEstimationNode": DepthEstimationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthEstimationNode": "Depth Estimation"
}