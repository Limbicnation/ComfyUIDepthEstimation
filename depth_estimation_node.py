import os
import numpy as np
import torch
from transformers import pipeline
from PIL import Image, ImageFilter, ImageOps
import folder_paths
from comfy.model_management import get_torch_device, get_free_memory
import gc
import logging
from typing import Tuple, List, Dict, Any, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DepthEstimation")

# Configure model paths
if not hasattr(folder_paths, "models_dir"):
    folder_paths.models_dir = os.path.join(folder_paths.base_path, "models")

# Register depth models path
DEPTH_DIR = "depth_anything"
folder_paths.folder_names_and_paths[DEPTH_DIR] = ([
    os.path.join(folder_paths.models_dir, DEPTH_DIR)
], folder_paths.supported_pt_extensions)

# Set models directory
MODELS_DIR = folder_paths.folder_names_and_paths[DEPTH_DIR][0][0]
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = MODELS_DIR

# Define all models mentioned in the README
DEPTH_MODELS = {
    "Depth-Anything-Small": "LiheYoung/depth-anything-small",
    "Depth-Anything-Base": "LiheYoung/depth-anything-base",
    "Depth-Anything-Large": "LiheYoung/depth-anything-large",
    "Depth-Anything-V2-Small": "LiheYoung/depth-anything-small-hf",
    "Depth-Anything-V2-Base": "LiheYoung/depth-anything-base-hf",
}

class DepthEstimationNode:
    """
    ComfyUI node for depth estimation using Depth Anything models.
    
    This node provides depth map generation from images using various Depth Anything models
    with configurable post-processing options like blur, median filtering, contrast enhancement,
    and gamma correction.
    """
    
    MEDIAN_SIZES = ["3", "5", "7", "9", "11"]
    
    def __init__(self):
        self.device = None
        self.depth_estimator = None
        self.current_model = None
        logger.info("Initialized DepthEstimationNode")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Define the input types for the node."""
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
    
    def cleanup(self) -> None:
        """Clean up resources and free VRAM."""
        try:
            if self.depth_estimator is not None:
                del self.depth_estimator
                self.depth_estimator = None
                self.current_model = None
                
                # Force CUDA cache clearing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                logger.info("Cleaned up model resources")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def ensure_model_loaded(self, model_name: str) -> None:
        """
        Ensures the correct model is loaded with proper VRAM management.
        
        Args:
            model_name: The name of the model to load
        
        Raises:
            RuntimeError: If the model fails to load
        """
        try:
            if model_name not in DEPTH_MODELS:
                raise ValueError(f"Unknown model: {model_name}. Available models: {list(DEPTH_MODELS.keys())}")
                
            model_path = DEPTH_MODELS[model_name]
            
            if self.depth_estimator is None or self.current_model != model_path:
                self.cleanup()
                
                if self.device is None:
                    self.device = get_torch_device()
                
                logger.info(f"Loading depth model: {model_name} on device {self.device}")
                
                # Determine device type for pipeline
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Use FP16 for CUDA devices to save VRAM
                dtype = torch.float16 if 'cuda' in str(self.device) else torch.float32
                
                # Check available VRAM before loading
                if torch.cuda.is_available():
                    free_vram = get_free_memory(self.device)
                    logger.info(f"Available VRAM before loading: {free_vram / (1024**3):.2f} GB")
                
                self.depth_estimator = pipeline(
                    "depth-estimation",
                    model=model_path,
                    device_map=device_type,
                    torch_dtype=dtype
                )
                
                # Ensure model is on the correct device
                if hasattr(self.depth_estimator, 'model'):
                    self.depth_estimator.model = self.depth_estimator.model.to(self.device)
                
                self.current_model = model_path
                logger.info(f"Successfully loaded {model_name}")
                
        except Exception as e:
            self.cleanup()
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def process_image(self, image: Union[torch.Tensor, np.ndarray]) -> Image.Image:
        """
        Converts input image to proper format for depth estimation.
        
        Args:
            image: Input image as tensor or numpy array
            
        Returns:
            PIL Image ready for depth estimation
        """
        if torch.is_tensor(image):
            image_np = (image.cpu().numpy()[0] * 255).astype(np.uint8)
        else:
            image_np = (image * 255).astype(np.uint8)
        
        if len(image_np.shape) == 3:
            if image_np.shape[-1] == 4:  # Handle RGBA images
                image_np = image_np[..., :3]
        elif len(image_np.shape) == 2:  # Handle grayscale images
            image_np = np.stack([image_np] * 3, axis=-1)
        
        return Image.fromarray(image_np)

    def estimate_depth(self, 
                     image: torch.Tensor, 
                     model_name: str, 
                     blur_radius: float = 2.0, 
                     median_size: str = "5", 
                     apply_auto_contrast: bool = True, 
                     apply_gamma: bool = True) -> Tuple[torch.Tensor]:
        """
        Estimates depth from input image with error handling and cleanup.
        
        Args:
            image: Input image tensor
            model_name: Name of the depth model to use
            blur_radius: Gaussian blur radius for smoothing
            median_size: Size of median filter for noise reduction
            apply_auto_contrast: Whether to enhance contrast automatically
            apply_gamma: Whether to apply gamma correction
            
        Returns:
            Tuple containing depth map tensor
            
        Raises:
            RuntimeError: If depth estimation fails
            ValueError: If invalid parameters are provided
        """
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
                depth_map = ((depth_map - depth_min) / (depth_max - depth_min) * 255.0)
            depth_map = depth_map.astype(np.uint8)
            
            # Create PIL image explicitly with L mode (grayscale)
            depth_pil = Image.fromarray(depth_map, mode='L')
            
            # Apply post-processing
            if blur_radius > 0:
                depth_pil = depth_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            if int(median_size) > 0:
                depth_pil = depth_pil.filter(ImageFilter.MedianFilter(size=int(median_size)))
            
            if apply_auto_contrast:
                depth_pil = ImageOps.autocontrast(depth_pil)
            
            if apply_gamma:
                depth_array = np.array(depth_pil).astype(np.float32) / 255.0
                mean_luminance = np.mean(depth_array)
                if mean_luminance > 0:
                    gamma = np.log(0.5) / np.log(mean_luminance)
                    # Use direct numpy operations for gamma correction
                    corrected = np.power(depth_array, 1.0/gamma) * 255.0
                    depth_pil = Image.fromarray(corrected.astype(np.uint8), mode='L')
            
            # Convert to tensor - explicitly handle as grayscale
            depth_array = np.array(depth_pil).astype(np.float32) / 255.0
            
            # Make it compatible with ComfyUI by creating a 3-channel image
            # Use proper reshaping to avoid dimension issues
            h, w = depth_array.shape
            depth_rgb = np.stack([depth_array] * 3, axis=-1)  # Create proper 3D array with shape (h, w, 3)
            
            depth_tensor = torch.from_numpy(depth_rgb).unsqueeze(0)
            
            if self.device is not None:
                depth_tensor = depth_tensor.to(self.device)
            
            return (depth_tensor,)
            
        except Exception as e:
            error_msg = f"Depth estimation failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def gamma_correction(self, img: Image.Image, gamma: float = 1.0) -> Image.Image:
        """Applies gamma correction to the image."""
        # Convert to numpy array
        img_array = np.array(img)
        
        # Apply gamma correction directly with numpy
        corrected = np.power(img_array.astype(np.float32) / 255.0, 1.0/gamma) * 255.0
        
        # Ensure uint8 type and create image with explicit mode
        return Image.fromarray(corrected.astype(np.uint8), mode='L')

# Node registration
NODE_CLASS_MAPPINGS = {
    "DepthEstimationNode": DepthEstimationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthEstimationNode": "Depth Estimation (V2)"
}