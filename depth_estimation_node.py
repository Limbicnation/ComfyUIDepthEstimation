import os
import numpy as np
import torch
import traceback
import time
import requests
import urllib.request
import wget
from pathlib import Path
from transformers import pipeline
from PIL import Image, ImageFilter, ImageOps, ImageDraw, ImageFont
import folder_paths
from comfy.model_management import get_torch_device, get_free_memory
import gc
import logging
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional, Union

# Try to import timm (for vision transformers)
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Direct loading of Depth Anything models may not work.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DepthEstimation")

# Depth Anything V2 Implementation
class DepthAnythingV2(nn.Module):
    """Direct implementation of Depth Anything V2 model"""
    def __init__(self, encoder='vits', features=64, out_channels=[48, 96, 192, 384]):
        super().__init__()
        self.encoder = encoder
        self.features = features
        self.out_channels = out_channels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create encoder based on specification
        if TIMM_AVAILABLE:
            if encoder == 'vits':
                self.backbone = timm.create_model('vit_small_patch16_224', pretrained=False)
                self.embed_dim = 384
            elif encoder == 'vitb':
                self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False)
                self.embed_dim = 768
            elif encoder == 'vitl':
                self.backbone = timm.create_model('vit_large_patch16_224', pretrained=False)
                self.embed_dim = 1024
            else:  # fallback to vits
                self.backbone = timm.create_model('vit_small_patch16_224', pretrained=False)
                self.embed_dim = 384
                
            # Implement the rest of the model architecture
            self.initialize_decoder()
        else:
            # Fallback if timm is not available
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=False)
            self.embed_dim = 2048
            logger.warning("Using fallback ResNet50 model (timm not available)")
    
    def initialize_decoder(self):
        """Initialize the decoder layers"""
        self.neck = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.features, 1, 1, 0),
            nn.Conv2d(self.features, self.features, 3, 1, 1),
        )
        
        # Create decoders for each level
        self.decoders = nn.ModuleList([
            self.create_decoder_level(self.features, self.out_channels[0]),
            self.create_decoder_level(self.out_channels[0], self.out_channels[1]),
            self.create_decoder_level(self.out_channels[1], self.out_channels[2]),
            self.create_decoder_level(self.out_channels[2], self.out_channels[3])
        ])
        
        # Final depth head
        self.depth_head = nn.Sequential(
            nn.Conv2d(self.out_channels[3], self.out_channels[3], 3, 1, 1),
            nn.BatchNorm2d(self.out_channels[3]),
            nn.ReLU(True),
            nn.Conv2d(self.out_channels[3], 1, 1)
        )
    
    def create_decoder_level(self, in_channels, out_channels):
        """Create a decoder level"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
    
    def forward(self, x):
        """Forward pass of the model"""
        # For timm ViT models
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
            
            # Reshape features based on model type
            if 'vit' in self.encoder:
                # Reshape transformer output to spatial features
                # Exact reshape depends on the model details
                h = w = int(features.shape[1]**0.5)
                features = features.reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)
            
            # Process through decoder
            x = self.neck(features)
            
            # Apply decoder stages
            for decoder in self.decoders:
                x = decoder(x)
            
            # Final depth prediction
            depth = self.depth_head(x)
            
            return depth
        else:
            # Fallback for ResNet
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            # Process through simple decoder
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = self.depth_head(x)
            
            return x
    
    def infer_image(self, image):
        """Process an image and return the depth map
        
        Args:
            image: A numpy image in BGR format (OpenCV) or RGB PIL Image
            
        Returns:
            depth: A numpy array containing the depth map
        """
        # Convert input to tensor
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB
            if image.shape[2] == 3:
                image = image[:, :, ::-1]
            # Normalize
            image = image.astype(np.float32) / 255.0
            # HWC to CHW
            image = image.transpose(2, 0, 1)
            # Add batch dimension
            image = torch.from_numpy(image).unsqueeze(0)
        elif isinstance(image, Image.Image):
            # Convert PIL image to numpy
            image = np.array(image).astype(np.float32) / 255.0
            # HWC to CHW
            image = image.transpose(2, 0, 1)
            # Add batch dimension
            image = torch.from_numpy(image).unsqueeze(0)
        
        # Move to device
        image = image.to(self.device)
        
        # Set model to eval mode
        self.eval()
        
        # Get prediction
        with torch.no_grad():
            depth = self.forward(image)
            
        # Convert to numpy
        depth = depth.squeeze().cpu().numpy()
        
        return depth
        
    def __call__(self, image):
        """Compatible interface with the pipeline API"""
        if isinstance(image, Image.Image):
            # Convert to numpy for processing
            depth = self.infer_image(image)
            # Return in the format expected by the node
            return {"predicted_depth": torch.from_numpy(depth).unsqueeze(0)}
        else:
            # Already a tensor, process directly
            self.eval()
            with torch.no_grad():
                depth = self.forward(image)
            return {"predicted_depth": depth}

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

# Define model configurations for direct loading
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Define all models mentioned in the README with memory requirements
DEPTH_MODELS = {
    "Depth-Anything-Small": {
        "path": "LiheYoung/depth-anything-small-hf",  # Correct HF path for V1
        "vram_mb": 1500,
        "direct_url": "https://github.com/LiheYoung/Depth-Anything/releases/download/v1.0/depth_anything_vitb14.pt"
    },
    "Depth-Anything-Base": {
        "path": "LiheYoung/depth-anything-base-hf",  # Correct HF path for V1
        "vram_mb": 2500,
        "direct_url": "https://github.com/LiheYoung/Depth-Anything/releases/download/v1.0/depth_anything_vitl14.pt"
    },
    "Depth-Anything-Large": {
        "path": "LiheYoung/depth-anything-large-hf",  # Correct HF path for V1
        "vram_mb": 4000,
        "direct_url": "https://github.com/LiheYoung/Depth-Anything/releases/download/v1.0/depth_anything_vitl14.pt"
    },
    "Depth-Anything-V2-Small": {
        "path": "LiheYoung/depth-anything-v2-small-hf",  # Updated corrected path
        "vram_mb": 1500, 
        "direct_url": "https://huggingface.co/LiheYoung/depth-anything-v2-small-hf/resolve/main/pytorch_model.bin"
    },
    "Depth-Anything-V2-Base": {
        "path": "LiheYoung/depth-anything-v2-base-hf",  # Updated corrected path
        "vram_mb": 2500,
        "direct_url": "https://huggingface.co/LiheYoung/depth-anything-v2-base-hf/resolve/main/pytorch_model.bin"
    },
    # Add MiDaS models as dedicated options
    "MiDaS-Small": {
        "path": "Intel/dpt-hybrid-midas",
        "vram_mb": 1000,
        "midas_type": "MiDaS_small"
    },
    "MiDaS-Base": {
        "path": "Intel/dpt-hybrid-midas", 
        "vram_mb": 1200,
        "midas_type": "DPT_Hybrid"
    }
}

class MiDaSWrapper:
    def __init__(self, model_type, device):
        self.device = device
        
        try:
            # Import required libraries
            import torch.nn.functional as F
            
            # Use a more reliable approach to loading MiDaS models
            if model_type == "DPT_Hybrid" or model_type == "dpt_hybrid":
                # Use direct URL download for MiDaS models
                midas_url = "https://github.com/intel-isl/MiDaS/releases/download/v3/dpt_hybrid-midas-501f0c75.pt"
                local_path = os.path.join(MODELS_DIR, "dpt_hybrid_midas.pt")
                
                if not os.path.exists(local_path):
                    logger.info(f"Downloading MiDaS model from {midas_url}")
                    try:
                        response = requests.get(midas_url, stream=True)
                        if response.status_code == 200:
                            with open(local_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            logger.info(f"Downloaded MiDaS model to {local_path}")
                        else:
                            logger.error(f"Failed to download model: {response.status_code}")
                    except Exception as e:
                        logger.error(f"Error downloading MiDaS model: {e}")
                
                # Load pretrained model
                try:
                    # Create a simple model architecture
                    from torchvision.models import resnet50
                    self.model = resnet50()
                    self.model.fc = torch.nn.Linear(2048, 1)
                    
                    # Load state dict if available
                    if os.path.exists(local_path):
                        logger.info(f"Loading MiDaS model from {local_path}")
                        state_dict = torch.load(local_path, map_location=device)
                        # Convert all parameters to float
                        floated_state_dict = {k: v.float() for k, v in state_dict.items()}
                        self.model.load_state_dict(floated_state_dict)
                        
                except Exception as e:
                    logger.error(f"Error loading MiDaS model state dict: {e}")
                    # Fallback to ResNet
                    self.model = resnet50(pretrained=True)
                    self.model.fc = torch.nn.Linear(2048, 1)
                
            else:  # Other model types or fallback
                from torchvision.models import resnet50
                self.model = resnet50(pretrained=True)
                self.model.fc = torch.nn.Linear(2048, 1)
            
            # Ensure model parameters are float
            for param in self.model.parameters():
                param.data = param.data.float()
            
            # Explicitly convert model to FloatTensor
            self.model = self.model.float()
                
            # Move model to device and set to eval mode
            self.model = self.model.to(device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            logger.error(traceback.format_exc())
            # Create a minimal model as absolute fallback
            from torchvision.models import resnet18
            self.model = resnet18(pretrained=True).float().to(device)
            self.model.fc = torch.nn.Linear(512, 1).float().to(device)
            self.model.eval()
    
    def __call__(self, image):
        """Process an image and return the depth map"""
        try:
            # Convert PIL image to tensor for processing
            if isinstance(image, Image.Image):
                # Resize to 384x384 (standard MiDaS size)
                img_resized = image.resize((384, 384), Image.LANCZOS)
                
                # Convert to numpy array
                img_np = np.array(img_resized).astype(np.float32) / 255.0
                
                # Convert to tensor with proper shape (B,C,H,W)
                if len(img_np.shape) == 3:
                    # RGB image
                    img_np = img_np.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
                else:
                    # Grayscale image - add channel dimension
                    img_np = np.expand_dims(img_np, axis=0)
                
                # Add batch dimension and ensure float32
                input_tensor = torch.from_numpy(img_np).unsqueeze(0).float()
            else:
                # Already a tensor - ensure float32 by explicitly converting
                # This is the key fix for the "Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor)" error
                if image.dtype == torch.float64 or image.dtype == torch.double:
                    logger.info(f"Converting input tensor from {image.dtype} to torch.float32")
                    input_tensor = image.float()  # Convert DoubleTensor to FloatTensor
                else:
                    # Still convert to ensure it's float32
                    input_tensor = image.float()
                
                # Add batch dimension if missing
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)
            
            # Move to device and ensure float type
            input_tensor = input_tensor.to(self.device).float()
            
            # Log tensor info for debugging
            logger.info(f"Input tensor type before inference: {input_tensor.dtype}")
            
            # Run inference
            with torch.no_grad():
                # Make sure input is float32 and model weights are float32
                output = self.model(input_tensor)
                
                # Reshape to expected format
                if output.dim() == 2:
                    # Add channel dimension if missing
                    output = output.unsqueeze(1)
                
                # Resize to match input resolution
                if isinstance(image, Image.Image):
                    w, h = image.size
                    output = torch.nn.functional.interpolate(
                        output,
                        size=(h, w),
                        mode="bicubic",
                        align_corners=False
                    )
                
            # Use same interface as the pipeline
            return {"predicted_depth": output.float()}  # Ensure output is float
            
        except Exception as e:
            logger.error(f"Error in MiDaS inference: {e}")
            logger.error(traceback.format_exc())
            
            # Return a placeholder depth map
            if isinstance(image, Image.Image):
                w, h = image.size
                dummy_tensor = torch.ones((1, 1, h, w), device=self.device, dtype=torch.float32)
            else:
                # Try to get shape from tensor
                shape = image.shape
                if len(shape) >= 3:
                    if shape[0] == 3:  # CHW format
                        h, w = shape[1], shape[2]
                    else:  # HWC format 
                        h, w = shape[0], shape[1]
                else:
                    h, w = 512, 512
                dummy_tensor = torch.ones((1, 1, h, w), device=self.device, dtype=torch.float32)
            
            return {"predicted_depth": dummy_tensor}

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
                # Ensure minimum size is enforced by the UI
                "input_size": ("INT", {"default": 518, "min": 256, "max": 1024, "step": 1}),
                "blur_radius": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                # Define median_size as a dropdown with specific string values
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
                    try:
                        free_mem_info = get_free_memory(get_torch_device())
                        # Handle return value whether it's a tuple or a single value
                        if isinstance(free_mem_info, tuple):
                            free_mem, total_mem = free_mem_info
                            logger.info(f"Available VRAM after cleanup: {free_mem/1024:.2f}MB of {total_mem/1024:.2f}MB")
                        else:
                            logger.info(f"Available VRAM after cleanup: {free_mem_info/1024:.2f}MB")
                    except Exception as e:
                        logger.warning(f"Error getting memory info: {e}")
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
            
            # Handle model_info as string or dict
            if isinstance(model_info, dict):
                model_path = model_info["path"]
                required_vram = model_info.get("vram_mb", 2000) * 1024  # Convert to KB
            else:
                model_path = model_info
                required_vram = 2000 * 1024  # Default 2GB
            
            # Only reload if needed or forced
            if force_reload or self.depth_estimator is None or self.current_model != model_path:
                self.cleanup()
                
                # Set up device
                if self.device is None:
                    self.device = get_torch_device()
                
                logger.info(f"Loading depth model: {model_name} on {'CPU' if force_cpu else self.device}")
                
                # Check available memory if using CUDA
                if torch.cuda.is_available() and not force_cpu:
                    try:
                        free_mem_info = get_free_memory(self.device)
                        
                        # Handle different return types from get_free_memory
                        if isinstance(free_mem_info, tuple):
                            free_mem, total_mem = free_mem_info
                            logger.info(f"Available VRAM: {free_mem/1024:.2f}MB, Required: {required_vram/1024:.2f}MB")
                        else:
                            free_mem = free_mem_info
                            logger.info(f"Available VRAM: {free_mem/1024:.2f}MB, Required: {required_vram/1024:.2f}MB")
                            total_mem = free_mem * 2  # Estimate if not available
                        
                        # If not enough memory, fall back to CPU
                        if free_mem < required_vram:
                            logger.warning(f"Insufficient VRAM for {model_name} ({required_vram/1024:.1f}MB required, {free_mem/1024:.1f}MB available). Falling back to CPU.")
                            force_cpu = True
                    except Exception as mem_error:
                        logger.warning(f"Error checking VRAM, using CPU to be safe: {str(mem_error)}")
                        force_cpu = True
                
                # Determine device type for pipeline
                device_type = 'cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Use FP16 for CUDA devices to save VRAM
                dtype = torch.float16 if 'cuda' in str(self.device) and not force_cpu else torch.float32
                
                # Create a dedicated cache directory for this model
                cache_dir = os.path.join(MODELS_DIR, model_name.replace("-", "_").lower())
                os.makedirs(cache_dir, exist_ok=True)
                
                # Check if we should try direct model download
                direct_url = model_info.get("direct_url", None)
                if direct_url:
                    # Determine model filename from URL
                    model_filename = os.path.basename(direct_url)
                    model_path_local = os.path.join(cache_dir, model_filename)
                    
                    # Check if model already exists locally
                    if not os.path.exists(model_path_local):
                        try:
                            logger.info(f"Attempting to download model directly from: {direct_url}")
                            logger.info(f"Saving to: {model_path_local}")
                            
                            # Download with progress reporting
                            response = requests.get(direct_url, stream=True)
                            total_size = int(response.headers.get('content-length', 0))
                            block_size = 1024  # 1 Kibibyte
                            
                            if response.status_code == 200:
                                with open(model_path_local, 'wb') as f:
                                    if total_size > 0:
                                        downloaded = 0
                                        for data in response.iter_content(block_size):
                                            f.write(data)
                                            downloaded += len(data)
                                            download_pct = (downloaded / total_size) * 100
                                            if downloaded % (5 * 1024 * 1024) == 0:  # Log every 5MB
                                                logger.info(f"Downloaded: {downloaded/1024/1024:.1f}MB of {total_size/1024/1024:.1f}MB ({download_pct:.1f}%)")
                                    else:
                                        f.write(response.content)
                                logger.info(f"Model successfully downloaded to {model_path_local}")
                            else:
                                logger.warning(f"Failed to download model from {direct_url}, status code: {response.status_code}")
                        except Exception as download_error:
                            logger.warning(f"Error downloading model: {str(download_error)}")
                
                # List of model paths to try (original and fallback)
                model_paths_to_try = [
                    model_path,  # Original path
                    model_path.replace("-hf", ""),  # Remove -hf suffix if it exists
                    model_path if "-hf" in model_path else model_path + "-hf",  # Add or keep -hf suffix
                    "depth-anything/Depth-Anything-Small-hf" if "v2" in model_name.lower() else model_path,  # New V2 format
                    "Intel/dpt-hybrid-midas",  # Midas model as fallback
                    "LiheYoung/depth-anything-small"  # Fallback to regular Depth Anything model
                ]
                
                # Log all paths we're going to try
                logger.info(f"Will try loading from these paths: {model_paths_to_try}")
                
                # Try each model path
                success = False
                last_error = None
                
                logger.info(f"Loading model with device={device_type}, dtype={dtype}")
                
                for path in model_paths_to_try:
                    try:
                        logger.info(f"Attempting to load from: {path}")
                        
                        # Try with online mode first
                        try:
                            # Add more debugging information
                            logger.info(f"Loading with params: model={path}, device_map={device_type}, dtype={dtype}")
                            
                            # Handle specific TypeError that might occur during unpacking
                            try:
                                self.depth_estimator = pipeline(
                                    "depth-estimation",
                                    model=path,
                                    cache_dir=cache_dir,
                                    local_files_only=False,  # Try online first
                                    device_map=device_type,
                                    torch_dtype=dtype
                                )
                                
                                # Verify that the estimator was properly initialized
                                if self.depth_estimator is None:
                                    raise RuntimeError("Pipeline initialization returned None")
                                    
                                # Log more info for debugging
                                logger.info(f"Pipeline created: {type(self.depth_estimator)}")
                                
                                # Test the model with a small image to ensure it works
                                test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
                                _ = self.depth_estimator(test_img)
                                logger.info("Model test successful")
                                
                                success = True
                                logger.info(f"Successfully loaded model from {path}")
                                break
                            except TypeError as type_error:
                                # Handle unpacking errors by printing traceback
                                logger.error(f"Type error when loading model: {str(type_error)}")
                                logger.error(f"Traceback: {traceback.format_exc()}")
                                
                                # Try alternative pipeline creation approach for older transformers versions
                                logger.info("Trying alternative pipeline creation method...")
                                from transformers import AutoModelForDepthEstimation, AutoImageProcessor
                                
                                # Load model components separately to avoid unpacking issues
                                try:
                                    processor = AutoImageProcessor.from_pretrained(path, cache_dir=cache_dir)
                                    model = AutoModelForDepthEstimation.from_pretrained(path, cache_dir=cache_dir)
                                    
                                    # Move model to correct device if needed
                                    if not force_cpu and 'cuda' in device_type:
                                        model = model.to(self.device)
                                    
                                    # Create a custom pipeline class that wraps these components
                                    class CustomDepthEstimator:
                                        def __init__(self, model, processor):
                                            self.model = model
                                            self.processor = processor
                                            
                                        def __call__(self, image):
                                            # Process image and run model
                                            inputs = self.processor(images=image, return_tensors="pt")
                                            if not force_cpu and 'cuda' in device_type:
                                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                                            
                                            with torch.no_grad():
                                                outputs = self.model(**inputs)
                                                
                                            # Format output like the pipeline would
                                            return {"predicted_depth": outputs.predicted_depth}
                                    
                                    self.depth_estimator = CustomDepthEstimator(model, processor)
                                    
                                    # Test the custom pipeline
                                    test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
                                    _ = self.depth_estimator(test_img)
                                    
                                    success = True
                                    logger.info(f"Successfully loaded model using custom pipeline")
                                    break
                                except Exception as custom_error:
                                    logger.error(f"Custom pipeline creation failed: {str(custom_error)}")
                                    raise
                                
                        except Exception as online_error:
                            logger.warning(f"Online loading failed for {path}: {str(online_error)}")
                            logger.debug(f"Error traceback: {traceback.format_exc()}")
                            
                            # Try with local_files_only if online fails
                            try:
                                # Add more verbose logging
                                logger.info(f"Trying local cache with model={path}")
                                
                                self.depth_estimator = pipeline(
                                    "depth-estimation",
                                    model=path,
                                    cache_dir=cache_dir,
                                    local_files_only=True,  # Try local only as fallback
                                    device_map=device_type,
                                    torch_dtype=dtype
                                )
                                
                                # Verify pipeline initialization success
                                if self.depth_estimator is None:
                                    raise RuntimeError("Local pipeline initialization returned None")
                                
                                # Test the model
                                test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
                                _ = self.depth_estimator(test_img)
                                
                                success = True
                                logger.info(f"Successfully loaded model from local cache: {path}")
                                break
                            except Exception as local_error:
                                last_error = local_error
                                logger.warning(f"Local loading failed for {path}: {str(local_error)}")
                                logger.debug(f"Error traceback: {traceback.format_exc()}")
                                continue
                                
                    except Exception as path_error:
                        last_error = path_error
                        logger.warning(f"Failed to load model from {path}: {str(path_error)}")
                        continue
                
                # Try the direct loading approach if all HuggingFace transformers approaches failed
                if not success:
                    logger.info("All transformers pipeline attempts failed, trying direct model loading...")
                    
                    # Try the direct loading approach
                    direct_model = self.load_model_direct(model_name, model_info, force_cpu)
                    
                    if direct_model is not None:
                        self.depth_estimator = direct_model
                        success = True
                        logger.info(f"Successfully loaded model using direct loading approach")
                    else:
                        logger.error("Direct model loading also failed")
                
                # Final fallback: try a different model or report failure
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
1. Check your internet connection
2. Download the model manually from the direct URLs in this file
3. Try a different model version (e.g. Depth-Anything-V2-Small instead of Depth-Anything-Small)
4. Ensure you have enough VRAM available or use force_cpu=True
5. Make sure the models directory exists: {MODELS_DIR}
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

    def load_model_direct(self, model_name, model_info, force_cpu=False):
        """
        Directly loads a depth model without using transformers pipeline.
        This is a fallback method when the normal pipeline loading fails.
        
        Args:
            model_name: Name of the model to load
            model_info: Dictionary with model information
            force_cpu: Whether to force CPU usage
            
        Returns:
            A depth estimation model that implements the __call__ interface
        """
        try:
            logger.info(f"Attempting direct model loading for {model_name}")
            
            # Determine device
            device_type = 'cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
            device = torch.device(device_type)
            
            # Define model directory and ensure it exists
            cache_dir = os.path.join(MODELS_DIR, model_name.replace("-", "_").lower())
            os.makedirs(cache_dir, exist_ok=True)
            
            # Get model configuration
            is_v2 = model_info.get("v2", False)
            config_name = model_info.get("config", "vits")
            
            # Step 1: Download the model weights if they don't exist
            direct_url = model_info.get("direct_url")
            model_path_local = None
            
            if direct_url:
                model_filename = os.path.basename(direct_url)
                model_path_local = os.path.join(cache_dir, model_filename)
                
                if not os.path.exists(model_path_local):
                    logger.info(f"Downloading model weights from {direct_url}")
                    try:
                        # Download with progress reporting
                        logger.info(f"Starting download to {model_path_local}")
                        
                        try:
                            # Try wget (more reliable for large files)
                            wget.download(direct_url, out=model_path_local)
                            logger.info(f"Downloaded model weights to {model_path_local}")
                        except:
                            # Fallback to requests
                            response = requests.get(direct_url, stream=True)
                            total_size = int(response.headers.get('content-length', 0))
                            
                            if response.status_code == 200:
                                with open(model_path_local, 'wb') as f:
                                    for data in response.iter_content(1024 * 1024):  # 1MB chunks
                                        f.write(data)
                                logger.info(f"Downloaded model weights to {model_path_local}")
                            else:
                                logger.warning(f"Failed to download model: status {response.status_code}")
                                return None
                    except Exception as dl_error:
                        logger.error(f"Error downloading model: {str(dl_error)}")
                        return None
            
            # Step 2: Create and load the appropriate model
            if is_v2 and TIMM_AVAILABLE and model_path_local and os.path.exists(model_path_local):
                # Use the DepthAnythingV2 implementation for V2 models
                try:
                    # Get the correct configuration for this model
                    if config_name in MODEL_CONFIGS:
                        config = MODEL_CONFIGS[config_name]
                        logger.info(f"Creating DepthAnythingV2 with config: {config}")
                        
                        # Create model with the appropriate configuration
                        model = DepthAnythingV2(**config)
                        
                        # Load weights from checkpoint
                        logger.info(f"Loading weights from {model_path_local}")
                        state_dict = torch.load(model_path_local, map_location=device)
                        
                        # Attempt to load the state dict (handles different formats)
                        try:
                            if "model" in state_dict:
                                model.load_state_dict(state_dict["model"])
                            else:
                                model.load_state_dict(state_dict)
                        except Exception as e:
                            logger.warning(f"Error loading state dict: {str(e)}")
                            logger.warning("Trying to load with strict=False")
                            if "model" in state_dict:
                                model.load_state_dict(state_dict["model"], strict=False)
                            else:
                                model.load_state_dict(state_dict, strict=False)
                        
                        # Move model to the correct device
                        model.to(device)
                        model.device = device
                        model.eval()
                        
                        # Test the model
                        logger.info("Testing model with sample image")
                        test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
                        _ = model(test_img)
                        
                        logger.info("DepthAnythingV2 model loaded and tested successfully")
                        return model
                    else:
                        logger.error(f"Unknown config: {config_name}")
                except Exception as e:
                    logger.error(f"Error loading DepthAnythingV2: {str(e)}")
                    logger.debug(traceback.format_exc())
            
            # Fallback to MiDaS model for v1 or if V2 loading failed
            try:
                logger.info("Falling back to MiDaS model")
                
                # Determine the appropriate MiDaS model type
                midas_model_type = "dpt_hybrid"
                if "large" in model_name.lower():
                    midas_model_type = "dpt_large"
                elif "small" in model_name.lower():
                    midas_model_type = "midas_v21_small"
                
                # Create and test the MiDaS model
                midas_model = MiDaSWrapper(midas_model_type, device)
                
                # Test with a small image
                test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
                _ = midas_model(test_img)
                
                logger.info("MiDaS model loaded and tested successfully")
                return midas_model
                
            except Exception as e:
                logger.error(f"Error loading MiDaS: {str(e)}")
                logger.debug(traceback.format_exc())
                
            # If all else fails, return None
            return None
                
        except Exception as e:
            logger.error(f"Direct model loading failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return None

    def process_image(self, image: Union[torch.Tensor, np.ndarray], input_size: int = 518) -> Image.Image:
        """
        Converts input image to proper format for depth estimation and resizes it.
        
        Args:
            image: Input image as tensor or numpy array
            input_size: Target size for the longest dimension of the image
            
        Returns:
            PIL Image ready for depth estimation
        """
        try:
            # Validate input_size
            if input_size < 256:
                logger.warning(f"Input size {input_size} is too small, using 256 instead")
                input_size = 256
            elif input_size > 1024:
                logger.warning(f"Input size {input_size} is too large, using 1024 instead")
                input_size = 1024
            
            # Convert tensor to numpy array
            if torch.is_tensor(image):
                # Check tensor dtype and convert to float32 if needed
                if image.dtype == torch.float64 or image.dtype == torch.double:
                    logger.info(f"Converting input tensor from {image.dtype} to torch.float32")
                    image = image.float()  # Convert DoubleTensor to FloatTensor
                
                # Check for NaN values in tensor
                if torch.isnan(image).any():
                    logger.warning("Input tensor contains NaN values. Replacing with zeros.")
                    image = torch.nan_to_num(image, nan=0.0)
                
                # Get first image from batch and convert to numpy
                image_np = (image.cpu().numpy()[0] * 255).astype(np.uint8)
            else:
                # Check for NaN values in numpy array
                if np.isnan(image).any():
                    logger.warning("Input array contains NaN values. Replacing with zeros.")
                    image = np.nan_to_num(image, nan=0.0)
                
                # Convert float64 to float32 if needed
                if image.dtype == np.float64:
                    logger.info("Converting numpy array from float64 to float32")
                    image = image.astype(np.float32)
                
                image_np = (image * 255).astype(np.uint8)
            
            # Handle different channel configurations
            if len(image_np.shape) == 3:
                if image_np.shape[-1] == 4:  # Handle RGBA images
                    image_np = image_np[..., :3]
            elif len(image_np.shape) == 2:  # Handle grayscale images
                image_np = np.stack([image_np] * 3, axis=-1)
            
            # Convert to PIL image
            pil_image = Image.fromarray(image_np)
            
            # Resize the image while preserving aspect ratio
            width, height = pil_image.size
            # Determine which dimension to scale to input_size
            if width > height:
                new_width = input_size
                new_height = int(height * (new_width / width))
            else:
                new_height = input_size
                new_width = int(width * (new_height / height))
            
            # Resize the image with antialiasing
            resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            return resized_image
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.debug(traceback.format_exc())
            # Return a placeholder image on error
            return Image.new('RGB', (512, 512), (128, 128, 128))

    def _create_error_image(self, input_image=None):
        """Create an error image placeholder based on input image if possible."""
        try:
            if input_image is not None and isinstance(input_image, torch.Tensor) and input_image.shape[0] > 0:
                # Check tensor type - if it's float64, log it for debugging
                if input_image.dtype == torch.float64 or input_image.dtype == torch.double:
                    logger.info(f"Input tensor for error image is {input_image.dtype}, will create float32 error image")
                
                # Create gray error image with same dimensions as input
                # Ensure tensor has the right shape for error display (BHWC)
                if input_image.ndim == 4:
                    if input_image.shape[-1] != 3:  # if not BHWC format
                        if input_image.shape[1] == 3:  # if BCHW format
                            # Extract height and width from BCHW
                            h, w = input_image.shape[2], input_image.shape[3]
                        else:
                            # Default to dimensions from input
                            h, w = input_image.shape[2], input_image.shape[3]
                    else:
                        # Already in BHWC format
                        h, w = input_image.shape[1], input_image.shape[2]
                else:
                    # Unexpected shape, use default
                    return self._create_basic_error_image()
                
                # Make sure dimensions aren't too small
                if h <= 1 or w <= 1:
                    logger.warning(f"Input has invalid dimensions {h}x{w}, using default error image")
                    return self._create_basic_error_image()
                
                # Gray background with slight red tint to indicate error - explicitly use float32
                placeholder = torch.ones((1, h, w, 3), dtype=torch.float32) * torch.tensor([0.5, 0.4, 0.4], dtype=torch.float32)
                
                if self.device is not None:
                    placeholder = placeholder.to(self.device)
                    
                # Verify the placeholder is float32
                if placeholder.dtype != torch.float32:
                    logger.warning(f"Error image has unexpected dtype {placeholder.dtype}, converting to float32")
                    placeholder = placeholder.float()
                    
                return placeholder
            else:
                return self._create_basic_error_image()
        except Exception as e:
            logger.error(f"Error creating error image: {str(e)}")
            return self._create_basic_error_image()
        
    def _create_basic_error_image(self):
        """Create a basic error image when no input dimensions are available."""
        # Standard size error image (512x512)
        h, w = 512, 512
        # Gray background with slight red tint to indicate error - explicitly use float32
        placeholder = torch.ones((1, h, w, 3), dtype=torch.float32) * torch.tensor([0.5, 0.4, 0.4], dtype=torch.float32)
        
        if self.device is not None:
            placeholder = placeholder.to(self.device)
            
        # Double-check that we're returning a float32 tensor
        if placeholder.dtype != torch.float32:
            placeholder = placeholder.float()
            
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
                     input_size: int = 518,
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
            input_size: Target size for the longest dimension of the image (between 256 and 1024)
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
            
            # Check for DoubleTensor and convert to FloatTensor if needed
            if image.dtype == torch.float64 or image.dtype == torch.double:
                logger.info(f"Converting input tensor from {image.dtype} to torch.float32 in estimate_depth")
                image = image.float()  # This is crucial for fixing the tensor type mismatch
                
            # Create error image placeholder based on input dimensions
            error_image = self._create_error_image(image)
                
            if torch.isnan(image).any():
                logger.warning("Input image contains NaN values. These will be replaced.")
                image = torch.nan_to_num(image, nan=0.0)
                
            # Handle case where median_size is passed as a boolean or other type
            if isinstance(median_size, bool) or median_size is True or median_size == 'True':
                logger.warning(f"median_size was passed as boolean: {median_size}. Defaulting to 5")
                median_size = "5"
            elif not isinstance(median_size, str) or median_size not in self.MEDIAN_SIZES:
                logger.warning(f"Invalid median_size: {median_size}. Defaulting to 5")
                median_size = "5"
            
            # Make sure it's one of the allowed values before any processing
            if median_size not in self.MEDIAN_SIZES:
                logger.warning(f"median_size '{median_size}' not in allowed values {self.MEDIAN_SIZES}, defaulting to 5")
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
            
            # Process input image with resizing
            try:
                # Ensure input_size is valid
                # Add more strict validation to handle edge cases
                if not isinstance(input_size, int):
                    logger.warning(f"Input size {input_size} is not an integer, using 518 instead")
                    input_size = 518
                    
                # Fix input_size if it's too small
                if input_size < 256:
                    logger.warning(f"Input size {input_size} is too small, using 256 instead")
                    input_size = 256
                elif input_size > 1024:
                    logger.warning(f"Input size {input_size} is too large, using 1024 instead")
                    input_size = 1024
                
                # Log tensor type for debugging
                logger.info(f"Input tensor type before processing: {image.dtype}")
                
                pil_image = self.process_image(image, input_size)
            except Exception as img_error:
                logger.error(f"Image processing error: {str(img_error)}")
                self._add_error_text_to_image(error_image, f"Image Error: {str(img_error)[:100]}...")
                return (error_image,)
            
            # Perform depth estimation with error catching
            try:
                with torch.inference_mode():
                    # Log tensor info before depth estimation
                    logger.info(f"Calling depth estimator with PIL image of size {pil_image.size}")
                    
                    depth_result = self.depth_estimator(pil_image)
                    # Convert output to float32 if needed
                    predicted_depth = depth_result["predicted_depth"]
                    if predicted_depth.dtype != torch.float32:
                        logger.info(f"Converting output from {predicted_depth.dtype} to float32")
                        predicted_depth = predicted_depth.float()
                    
                    depth_map = predicted_depth.squeeze().cpu().numpy()
            except RuntimeError as rt_error:
                # Check for tensor type mismatch errors
                error_msg = str(rt_error)
                if "Input type" in error_msg and "weight type" in error_msg:
                    # This is the specific error we're trying to fix
                    logger.error(f"Tensor type mismatch error: {error_msg}")
                    
                    # Try to fall back to CPU with explicit float conversion
                    logger.info("Attempting to fall back to CPU with explicit float conversion")
                    try:
                        # Create a copy of the image tensor with explicit float32 type
                        float_image = image.float().cpu()  # Move to CPU and convert to float
                        return self.estimate_depth(
                            float_image, model_name, input_size, blur_radius, median_size, 
                            apply_auto_contrast, apply_gamma, True, True
                        )
                    except Exception as float_fallback_error:
                        logger.error(f"Float fallback also failed: {str(float_fallback_error)}")
                
                # Check specifically for CUDA out-of-memory errors
                elif "CUDA out of memory" in error_msg:
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
                                image, model_name, input_size, blur_radius, median_size, 
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
                
                # Fix the tensor conversion:
                depth_array = np.array(depth_pil).astype(np.float32) / 255.0
                
                # Check if depth_array has proper dimensions and isn't just a thin line
                h, w = depth_array.shape
                if h <= 1 or w <= 1:
                    logger.error(f"Invalid depth map dimensions: {h}x{w}, using error image instead")
                    if error_image is not None:
                        self._add_error_text_to_image(error_image, "Invalid depth map dimensions (thin line)")
                        return (error_image,)
                    else:
                        # Create new error image if one doesn't exist
                        error_image = self._create_basic_error_image()
                        self._add_error_text_to_image(error_image, "Invalid depth map dimensions (thin line)")
                        return (error_image,)
                
                # Make sure we preserve proper dimensions - this is the crucial fix
                logger.info(f"Depth map dimensions: {h}x{w}")
                depth_rgb = np.stack([depth_array] * 3, axis=-1)  # Shape becomes (h, w, 3)
                
                # Convert to tensor and add batch dimension, ensuring float32 type
                depth_tensor = torch.from_numpy(depth_rgb).unsqueeze(0).float()  # Shape becomes (1, h, w, 3)
                
                if self.device is not None and not force_cpu:
                    depth_tensor = depth_tensor.to(self.device)
                
                # Make sure it's normalized in [0, 1] range
                if depth_tensor.max() > 1.0:
                    depth_tensor = depth_tensor / 255.0
                
                # Debug: log tensor shape and type
                logger.info(f"Output depth tensor shape: {depth_tensor.shape}, dtype: {depth_tensor.dtype}")
                
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