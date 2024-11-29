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
    MEDIAN_SIZES = ["3", "5", "7", "9", "11"]

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
                "median_size": (cls.MEDIAN_SIZES, {"default": "5"}),
                "apply_auto_contrast": ("BOOLEAN", {"default": True}),
                "apply_gamma": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_depth"
    CATEGORY = "image/depth"

    def ensure_model_loaded(self, model_name):
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
        try:
            # Validate median_size
            if median_size not in self.MEDIAN_SIZES:
                raise ValueError(f"Invalid median_size. Must be one of {self.MEDIAN_SIZES}")
            
            median_size_int = int(median_size)
            self.ensure_model_loaded(model_name)
            
            # Handle tensor conversion
            if torch.is_tensor(image):
                image_np = image.cpu().numpy()[0]  # Remove batch dimension
            else:
                image_np = image

            # Ensure proper RGB format and scaling
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
                
            # Convert to RGB if necessary
            if len(image_np.shape) == 3 and image_np.shape[-1] == 4:
                image_np = image_np[..., :3]
            elif len(image_np.shape) == 2:
                # Convert grayscale to RGB
                image_np = np.stack([image_np] * 3, axis=-1)
                
            # Convert to PIL for processing
            pil_image = Image.fromarray(image_np)
            
            # Get depth map
            depth_result = self.depth_estimator(pil_image)
            depth_map = depth_result["predicted_depth"]
            
            # Convert tensor to numpy and ensure correct dimensions
            if torch.is_tensor(depth_map):
                depth_map = depth_map.squeeze().cpu().numpy()
                
            # Ensure depth_map is 2D
            depth_map = depth_map.squeeze()
                
            # Normalize depth values to 0-255 range
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max > depth_min:
                depth_map = ((depth_map - depth_min) * (255.0 / (depth_max - depth_min)))
            else:
                depth_map = np.zeros_like(depth_map)
            
            depth_map = depth_map.astype(np.uint8)
            
            # Convert to PIL Image
            depth_map = Image.fromarray(depth_map, mode='L')  # Convert as grayscale
            
            # Apply post-processing
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
            
            # Convert back to tensor format
            depth_array = np.array(depth_map).astype(np.float32) / 255.0
            
            # Convert single channel to 3 channels
            depth_array = np.stack([depth_array] * 3, axis=-1)
            
            # Add batch dimension
            depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)
            
            # Move tensor to the correct device
            depth_tensor = depth_tensor.to(self.device)
            
            return (depth_tensor,)
            
        except Exception as e:
            raise RuntimeError(f"Depth estimation failed: {str(e)}")

    def gamma_correction(self, img, gamma=1.0):
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