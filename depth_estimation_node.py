import os
import numpy as np
import torch
from transformers import pipeline
from PIL import Image, ImageFilter, ImageOps
import folder_paths
from comfy.model_management import get_torch_device, get_free_memory
import gc
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DepthEstimation")

# Configure model caching directory
MODELS_DIR = os.path.join(folder_paths.get_folder_paths("models")[0], "depth_anything")
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = MODELS_DIR

DEPTH_MODELS = {
    "Depth-Anything-V2-Small": "LiheYoung/depth-anything-small-hf",
    "Depth-Anything-V2-Base": "LiheYoung/depth-anything-base-hf",
}

class DepthEstimationNode:
    """ComfyUI node for depth estimation using Depth Anything models."""
    
    MEDIAN_SIZES = ["3", "5", "7", "9", "11"]
    
    def __init__(self):
        self.device = None
        self.depth_estimator = None
        self.current_model = None
        logger.info("Initialized DepthEstimationNode")
    
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
    CATEGORY = "depth"
    
    def cleanup(self):
        """Clean up resources and VRAM."""
        if self.depth_estimator is not None:
            del self.depth_estimator
            self.depth_estimator = None
            self.current_model = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleaned up model resources")
    
    def ensure_model_loaded(self, model_name):
        """Ensures the correct model is loaded with proper VRAM management."""
        try:
            model_path = DEPTH_MODELS[model_name]
            
            if self.depth_estimator is None or self.current_model != model_path:
                self.cleanup()
                
                if self.device is None:
                    self.device = get_torch_device()
                
                logger.info(f"Loading depth model: {model_name} on device {self.device}")
                
                # Use FP16 for CUDA devices to save VRAM
                dtype = torch.float16 if 'cuda' in self.device else torch.float32
                
                self.depth_estimator = pipeline(
                    "depth-estimation",
                    model=model_path,
                    device=self.device,
                    torch_dtype=dtype
                )
                self.current_model = model_path
                logger.info(f"Successfully loaded {model_name}")
                
        except Exception as e:
            self.cleanup()
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def process_image(self, image):
        """Converts input image to proper format for depth estimation."""
        if torch.is_tensor(image):
            image_np = (image.cpu().numpy()[0] * 255).astype(np.uint8)
        else:
            image_np = (image * 255).astype(np.uint8)
        
        if len(image_np.shape) == 3:
            if image_np.shape[-1] == 4:
                image_np = image_np[..., :3]
        elif len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        
        return Image.fromarray(image_np)

    def estimate_depth(self, image, model_name, blur_radius=2.0, median_size="5", 
                      apply_auto_contrast=True, apply_gamma=True):
        """Estimates depth from input image with error handling and cleanup."""
        try:
            if median_size not in self.MEDIAN_SIZES:
                raise ValueError(f"Invalid median_size. Must be one of {self.MEDIAN_SIZES}")
            
            self.ensure_model_loaded(model_name)
            pil_image = self.process_image(image)
            
            with torch.inference_mode():
                depth_result = self.depth_estimator(pil_image)
                depth_map = depth_result["predicted_depth"].squeeze().cpu().numpy()
            
            # Normalize depth values
            depth_min, depth_max = depth_map.min(), depth_map.max()
            if depth_max > depth_min:
                depth_map = ((depth_map - depth_min) * (255.0 / (depth_max - depth_min)))
            depth_map = depth_map.astype(np.uint8)
            depth_map = Image.fromarray(depth_map, mode='L')
            
            # Apply post-processing
            if blur_radius > 0:
                depth_map = depth_map.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            if int(median_size) > 0:
                depth_map = depth_map.filter(ImageFilter.MedianFilter(size=int(median_size)))
            
            if apply_auto_contrast:
                depth_map = ImageOps.autocontrast(depth_map)
            
            if apply_gamma:
                depth_array = np.array(depth_map).astype(np.float32) / 255.0
                mean_luminance = np.mean(depth_array)
                if mean_luminance > 0:
                    gamma = np.log(0.5) / np.log(mean_luminance)
                    depth_map = self.gamma_correction(depth_map, gamma)
            
            # Convert to tensor
            depth_array = np.array(depth_map).astype(np.float32) / 255.0
            depth_array = np.stack([depth_array] * 3, axis=-1)
            depth_tensor = torch.from_numpy(depth_array).unsqueeze(0).to(self.device)
            
            return (depth_tensor,)
            
        except Exception as e:
            error_msg = f"Depth estimation failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def gamma_correction(self, img, gamma=1.0):
        """Applies gamma correction to the image."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], np.uint8)
        return Image.fromarray(np.array(img)).point(lambda x: table[x])

# Node registration
NODE_CLASS_MAPPINGS = {
    "DepthEstimationNode": DepthEstimationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthEstimationNode": "Depth Estimation (V2)"
}