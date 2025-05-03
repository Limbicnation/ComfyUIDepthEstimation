import os
import numpy as np
import torch
import traceback
import time
from transformers import pipeline
from PIL import Image, ImageFilter, ImageOps, ImageDraw, ImageFont
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

# Define all models mentioned in the README with memory requirements
DEPTH_MODELS = {
    "Depth-Anything-Small": {
        "path": "LiheYoung/depth-anything-small",
        "vram_mb": 1500
    },
    "Depth-Anything-Base": {
        "path": "LiheYoung/depth-anything-base",
        "vram_mb": 2500
    },
    "Depth-Anything-Large": {
        "path": "LiheYoung/depth-anything-large",
        "vram_mb": 4000
    },
    "Depth-Anything-V2-Small": {
        "path": "LiheYoung/depth-anything-small-hf",
        "vram_mb": 1500
    },
    "Depth-Anything-V2-Base": {
        "path": "LiheYoung/depth-anything-base-hf",
        "vram_mb": 2500
    },
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
            },
            "optional": {
                "force_reload": ("BOOLEAN", {"default": False}),
                "force_cpu": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_depth"
    CATEGORY = "depth"
    
    def cleanup(self) -> None:
        """Clean up resources and free VRAM."""
        try:
            if self.depth_estimator is not None:
                # Save model name before deletion for logging
                model_name = self.current_model
                
                # Delete the estimator
                del self.depth_estimator
                self.depth_estimator = None
                self.current_model = None
                
                # Force CUDA cache clearing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                logger.info(f"Cleaned up model resources for {model_name}")
                
                # Log available memory after cleanup if CUDA is available
                if torch.cuda.is_available():
                    free_mem, total_mem = get_free_memory(get_torch_device())
                    logger.info(f"Available VRAM after cleanup: {free_mem/1024:.2f}MB of {total_mem/1024:.2f}MB")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
            logger.debug(traceback.format_exc())
    
    def ensure_model_loaded(self, model_name: str, force_reload: bool = False, force_cpu: bool = False) -> None:
        """
        Ensures the correct model is loaded with proper VRAM management and fallback options.
        
        Args:
            model_name: The name of the model to load
            force_reload: If True, reload the model even if it's already loaded
            force_cpu: If True, force loading on CPU regardless of GPU availability
        
        Raises:
            RuntimeError: If the model fails to load after all fallback attempts
        """
        try:
            if model_name not in DEPTH_MODELS:
                available_models = list(DEPTH_MODELS.keys())
                if len(available_models) > 0:
                    fallback_model = available_models[0]
                    logger.warning(f"Unknown model: {model_name}. Falling back to {fallback_model}")
                    model_name = fallback_model
                else:
                    raise ValueError(f"No depth models available. Please check your installation.")
                
            model_info = DEPTH_MODELS[model_name]
            model_path = model_info["path"]
            
            # Only reload if needed or forced
            if force_reload or self.depth_estimator is None or self.current_model != model_path:
                self.cleanup()
                
                # Set up device
                if self.device is None:
                    self.device = get_torch_device()
                
                logger.info(f"Loading depth model: {model_name} on {'CPU' if force_cpu else self.device}")
                
                # Check available memory if using CUDA
                if torch.cuda.is_available() and not force_cpu:
                    free_mem, total_mem = get_free_memory(self.device)
                    required_mem = model_info.get("vram_mb", 2000) * 1024  # Convert to KB
                    
                    logger.info(f"Available VRAM: {free_mem/1024:.2f}MB, Required: {required_mem/1024:.2f}MB")
                    
                    # If not enough memory, fall back to CPU
                    if free_mem < required_mem:
                        logger.warning(f"Insufficient VRAM for {model_name} ({required_mem/1024:.1f}MB required, {free_mem/1024:.1f}MB available). Falling back to CPU.")
                        force_cpu = True
                
                # Determine device type for pipeline
                device_type = 'cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Use FP16 for CUDA devices to save VRAM
                dtype = torch.float16 if 'cuda' in str(self.device) and not force_cpu else torch.float32
                
                # Create a dedicated cache directory for this model
                cache_dir = os.path.join(MODELS_DIR, model_name.replace("-", "_").lower())
                os.makedirs(cache_dir, exist_ok=True)
                
                # List of model paths to try (original and fallback)
                model_paths_to_try = [
                    model_path,  # Original path
                    model_path + "-hf",  # Try with -hf suffix
                    model_path.replace("depth-anything", "depth-anything-hf")  # Alternative format
                ]
                
                # Try each model path
                success = False
                last_error = None
                
                logger.info(f"Loading model with device={device_type}, dtype={dtype}")
                
                for path in model_paths_to_try:
                    try:
                        logger.info(f"Attempting to load from: {path}")
                        
                        # Try with online mode first
                        try:
                            self.depth_estimator = pipeline(
                                "depth-estimation",
                                model=path,
                                cache_dir=cache_dir,
                                local_files_only=False,  # Try online first
                                device_map=device_type,
                                torch_dtype=dtype
                            )
                            success = True
                            logger.info(f"Successfully loaded model from {path}")
                            break
                        except Exception as online_error:
                            logger.warning(f"Online loading failed for {path}: {str(online_error)}")
                            
                            # Try with local_files_only if online fails
                            try:
                                self.depth_estimator = pipeline(
                                    "depth-estimation",
                                    model=path,
                                    cache_dir=cache_dir,
                                    local_files_only=True,  # Try local only as fallback
                                    device_map=device_type,
                                    torch_dtype=dtype
                                )
                                success = True
                                logger.info(f"Successfully loaded model from local cache: {path}")
                                break
                            except Exception as local_error:
                                last_error = local_error
                                logger.warning(f"Local loading failed for {path}: {str(local_error)}")
                                continue
                                
                    except Exception as path_error:
                        last_error = path_error
                        logger.warning(f"Failed to load model from {path}: {str(path_error)}")
                        continue
                
                if not success:
                    # If all attempts failed, try a different model
                    if model_name != "Depth-Anything-V2-Small" and "Depth-Anything-V2-Small" in DEPTH_MODELS:
                        logger.warning(f"Failed to load {model_name}, trying Depth-Anything-V2-Small as fallback")
                        try:
                            # Increase chances of success with CPU
                            return self.ensure_model_loaded("Depth-Anything-V2-Small", True, True)
                        except Exception as fallback_error:
                            logger.error(f"Fallback model also failed: {str(fallback_error)}")
                    
                    # If still failing, show helpful message with instructions
                    error_msg = f"""
Failed to load model {model_name} after trying multiple sources.
Last error: {str(last_error)}

Try these solutions:
1. Run 'huggingface-cli login' in your terminal to authenticate
2. Check your internet connection
3. Try a different model version (e.g. Depth-Anything-V2-Small instead of Depth-Anything-Small)
4. Ensure you have enough VRAM available or use force_cpu=True
"""
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Ensure model is on the correct device
                if not force_cpu and hasattr(self.depth_estimator, 'model'):
                    self.depth_estimator.model = self.depth_estimator.model.to(self.device)
                
                self.current_model = model_path
                
        except Exception as e:
            self.cleanup()
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise RuntimeError(error_msg)

    def process_image(self, image: Union[torch.Tensor, np.ndarray]) -> Image.Image:
        """
        Converts input image to proper format for depth estimation.
        
        Args:
            image: Input image as tensor or numpy array
            
        Returns:
            PIL Image ready for depth estimation
        """
        try:
            if torch.is_tensor(image):
                # Check for NaN values in tensor
                if torch.isnan(image).any():
                    logger.warning("Input tensor contains NaN values. Replacing with zeros.")
                    image = torch.nan_to_num(image, nan=0.0)
                
                image_np = (image.cpu().numpy()[0] * 255).astype(np.uint8)
            else:
                # Check for NaN values in numpy array
                if np.isnan(image).any():
                    logger.warning("Input array contains NaN values. Replacing with zeros.")
                    image = np.nan_to_num(image, nan=0.0)
                
                image_np = (image * 255).astype(np.uint8)
            
            if len(image_np.shape) == 3:
                if image_np.shape[-1] == 4:  # Handle RGBA images
                    image_np = image_np[..., :3]
            elif len(image_np.shape) == 2:  # Handle grayscale images
                image_np = np.stack([image_np] * 3, axis=-1)
            
            return Image.fromarray(image_np)
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.debug(traceback.format_exc())
            # Return a placeholder image on error
            return Image.new('RGB', (512, 512), (128, 128, 128))

    def _create_error_image(self, input_image=None):
        """Create an error image placeholder based on input image if possible."""
        try:
            if input_image is not None and isinstance(input_image, torch.Tensor) and input_image.shape[0] > 0:
                # Create gray error image with same dimensions as input
                h, w = input_image.shape[2], input_image.shape[3]
                # Gray background with slight red tint to indicate error
                placeholder = torch.ones((1, h, w, 3), dtype=torch.float32) * torch.tensor([0.5, 0.4, 0.4])
                
                if self.device is not None:
                    placeholder = placeholder.to(self.device)
                    
                return placeholder
            else:
                return self._create_basic_error_image()
        except Exception:
            return self._create_basic_error_image()
        
    def _create_basic_error_image(self):
        """Create a basic error image when no input dimensions are available."""
        # Standard size error image (512x512)
        h, w = 512, 512
        # Gray background with slight red tint to indicate error
        placeholder = torch.ones((1, h, w, 3), dtype=torch.float32) * torch.tensor([0.5, 0.4, 0.4])
        
        if self.device is not None:
            placeholder = placeholder.to(self.device)
            
        return placeholder
    
    def _add_error_text_to_image(self, image_tensor, error_text):
        """Add error text to the image tensor for visual feedback."""
        try:
            # Convert tensor to PIL for text rendering
            if image_tensor is None:
                return
                
            temp_img = self._tensor_to_pil(image_tensor)
            
            # Draw error text
            draw = ImageDraw.Draw(temp_img)
            
            # Try to get a font, fall back to default if needed
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Split text into multiple lines if too long
            lines = []
            words = error_text.split()
            current_line = words[0] if words else "Error"
            
            for word in words[1:]:
                if len(current_line + " " + word) < 50:
                    current_line += " " + word
                else:
                    lines.append(current_line)
                    current_line = word
                    
            lines.append(current_line)
            
            # Draw title
            draw.text((10, 10), "Depth Estimation Error", fill=(255, 50, 50), font=font)
            
            # Draw error message
            y_position = 40
            for line in lines:
                draw.text((10, y_position), line, fill=(255, 255, 255), font=font)
                y_position += 25
            
            # Convert back to tensor
            result = self._pil_to_tensor(temp_img)
            
            # Copy to original tensor if shapes match
            if image_tensor.shape == result.shape:
                image_tensor.copy_(result)
            return image_tensor
                
        except Exception as e:
            logger.error(f"Error adding text to error image: {e}")
            return image_tensor
            
    def _tensor_to_pil(self, tensor):
        """Convert a tensor to PIL Image."""
        if tensor.shape[0] == 1:  # Batch size 1
            img_np = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(img_np)
        return Image.new('RGB', (512, 512), color=(128, 100, 100))
        
    def _pil_to_tensor(self, pil_img):
        """Convert PIL Image back to tensor."""
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np).unsqueeze(0)
        
        if self.device is not None:
            tensor = tensor.to(self.device)
            
        return tensor

    def estimate_depth(self, 
                     image: torch.Tensor, 
                     model_name: str, 
                     blur_radius: float = 2.0, 
                     median_size: str = "5", 
                     apply_auto_contrast: bool = True, 
                     apply_gamma: bool = True,
                     force_reload: bool = False,
                     force_cpu: bool = False) -> Tuple[torch.Tensor]:
        """
        Estimates depth from input image with error handling and cleanup.
        
        Args:
            image: Input image tensor
            model_name: Name of the depth model to use
            blur_radius: Gaussian blur radius for smoothing
            median_size: Size of median filter for noise reduction
            apply_auto_contrast: Whether to enhance contrast automatically
            apply_gamma: Whether to apply gamma correction
            force_reload: Whether to force reload the model
            force_cpu: Whether to force using CPU for inference
            
        Returns:
            Tuple containing depth map tensor
        """
        error_image = None
        start_time = time.time()
        
        try:
            # Validate inputs
            if image is None or image.numel() == 0:
                raise ValueError("Empty or null input image")
                
            if image.ndim != 4:
                raise ValueError(f"Expected 4D tensor for image, got {image.ndim}D.")
                
            # Create error image placeholder based on input dimensions
            error_image = self._create_error_image(image)
                
            if torch.isnan(image).any():
                logger.warning("Input image contains NaN values. These will be replaced.")
                image = torch.nan_to_num(image, nan=0.0)
                
            if median_size not in self.MEDIAN_SIZES:
                logger.warning(f"Invalid median_size: {median_size}. Defaulting to 5")
                median_size = "5"
            
            # Load model with fallback strategy - wrapped in try-except
            try:
                self.ensure_model_loaded(model_name, force_reload, force_cpu)
            except Exception as model_error:
                # Special handling for model loading errors - common issue
                error_msg = f"Failed to load model '{model_name}': {str(model_error)}"
                logger.error(error_msg)
                # Add error text to error image
                self._add_error_text_to_image(error_image, f"Model Error: {str(model_error)[:100]}...")
                return (error_image,)
            
            # Process input image
            try:
                pil_image = self.process_image(image)
            except Exception as img_error:
                logger.error(f"Image processing error: {str(img_error)}")
                self._add_error_text_to_image(error_image, f"Image Error: {str(img_error)[:100]}...")
                return (error_image,)
            
            # Perform depth estimation with error catching
            try:
                with torch.inference_mode():
                    depth_result = self.depth_estimator(pil_image)
                    depth_map = depth_result["predicted_depth"].squeeze().cpu().numpy()
            except RuntimeError as rt_error:
                # Check specifically for CUDA out-of-memory errors
                if "CUDA out of memory" in str(rt_error):
                    error_msg = (
                        f"CUDA out of memory while processing depth map. "
                        f"Try using a smaller model or reducing image size."
                    )
                    logger.error(error_msg)
                    
                    # Try to fall back to CPU if we hit OOM
                    if not force_cpu:
                        logger.info("Attempting to fall back to CPU due to CUDA OOM error")
                        try:
                            return self.estimate_depth(
                                image, model_name, blur_radius, median_size, 
                                apply_auto_contrast, apply_gamma, True, True
                            )
                        except Exception as cpu_fallback_error:
                            logger.error(f"CPU fallback also failed: {str(cpu_fallback_error)}")
                            
                    self._add_error_text_to_image(error_image, "CUDA Out of Memory. Try a smaller model.")
                    return (error_image,)
                else:
                    # Other runtime errors
                    error_msg = f"Runtime error during depth estimation: {str(rt_error)}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    self._add_error_text_to_image(error_image, f"Runtime Error: {str(rt_error)[:100]}...")
                    return (error_image,)
            except Exception as e:
                # General exceptions
                error_msg = f"Depth estimation failed: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                self._add_error_text_to_image(error_image, f"Error: {str(e)[:100]}...")
                return (error_image,)
            
            # Check for NaN values in depth map
            if np.isnan(depth_map).any():
                logger.warning("Depth map contains NaN values. Replacing with zeros.")
                depth_map = np.nan_to_num(depth_map, nan=0.0)
            
            # Continue with the normal depth map processing
            try:
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
                h, w = depth_array.shape
                depth_rgb = np.stack([depth_array] * 3, axis=-1)  # Create proper 3D array with shape (h, w, 3)
                
                depth_tensor = torch.from_numpy(depth_rgb).unsqueeze(0)
                
                if self.device is not None:
                    depth_tensor = depth_tensor.to(self.device)
                
                processing_time = time.time() - start_time
                logger.info(f"Depth processing completed in {processing_time:.2f} seconds")
                
                return (depth_tensor,)
                
            except Exception as post_error:
                error_msg = f"Error during depth map post-processing: {str(post_error)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                self._add_error_text_to_image(error_image, f"Post-processing Error: {str(post_error)[:100]}...")
                return (error_image,)
                
        except Exception as e:
            # Catch-all for any other exceptions
            error_msg = f"Depth estimation failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            
            # If error_image hasn't been created yet, create a basic one
            if error_image is None:
                error_image = self._create_basic_error_image()
                
            self._add_error_text_to_image(error_image, f"Unexpected Error: {str(e)[:100]}...")
            return (error_image,)
        finally:
            # Always clean up resources
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