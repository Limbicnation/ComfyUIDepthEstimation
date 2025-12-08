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
from dataclasses import dataclass

# Custom ComfyUI type definitions for camera parameters
CAMERA_EXTRINSICS = "CAMERA_EXTRINSICS"
CAMERA_INTRINSICS = "CAMERA_INTRINSICS"

# Try to import timm (for vision transformers)
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Direct loading of Depth Anything models may not work.")

# Get logger instance (basicConfig is called in __init__.py)
logger = logging.getLogger("DepthEstimation")

# Import DA3 availability status from the package's __init__
from . import DA3_AVAILABLE

# Conditionally import Depth Anything V3 if available
# Use defensive import guard to handle edge cases where DA3_AVAILABLE check passes
# but the actual import still fails (e.g., corrupted install, version mismatch)
if DA3_AVAILABLE:
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as e:
        DA3_AVAILABLE = False
        logger.warning(f"DA3 import failed despite availability check: {e}. DA3 models disabled.")

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

# Register depth models path - support multiple possible directory structures
DEPTH_DIR = "depth_anything"
DEPTH_ANYTHING_DIR = "depthanything"

# Check which directory structure exists
possible_paths = [
    os.path.join(folder_paths.models_dir, DEPTH_DIR),
    os.path.join(folder_paths.models_dir, DEPTH_ANYTHING_DIR),
    os.path.join(folder_paths.models_dir, DEPTH_ANYTHING_DIR, DEPTH_DIR),
    os.path.join(folder_paths.models_dir, "checkpoints", DEPTH_DIR),
    os.path.join(folder_paths.models_dir, "checkpoints", DEPTH_ANYTHING_DIR),
]

# Filter to only paths that exist
existing_paths = [p for p in possible_paths if os.path.exists(p)]
if not existing_paths:
    # If none exists, create the default one
    existing_paths = [os.path.join(folder_paths.models_dir, DEPTH_DIR)]
    os.makedirs(existing_paths[0], exist_ok=True)
    logger.info(f"Created model directory: {existing_paths[0]}")

# Log all found paths for debugging
logger.info(f"Found depth model directories: {existing_paths}")

# Register all possible paths for model loading
folder_paths.folder_names_and_paths[DEPTH_DIR] = (existing_paths, folder_paths.supported_pt_extensions)

# Set primary models directory to the first available path
MODELS_DIR = existing_paths[0]
logger.info(f"Using primary models directory: {MODELS_DIR}")

# Set Hugging Face cache to the models directory to ensure models are saved there
os.environ["TRANSFORMERS_CACHE"] = MODELS_DIR
os.environ["HF_HOME"] = MODELS_DIR

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
        "direct_url": "https://github.com/LiheYoung/Depth-Anything/releases/download/v1.0/depth_anything_vitb14.pt",
        "model_type": "v1",
        "encoder": "vitb"
    },
    "Depth-Anything-Base": {
        "path": "LiheYoung/depth-anything-base-hf",  # Correct HF path for V1
        "vram_mb": 2500,
        "direct_url": "https://github.com/LiheYoung/Depth-Anything/releases/download/v1.0/depth_anything_vitl14.pt",
        "model_type": "v1",
        "encoder": "vitl"
    },
    "Depth-Anything-Large": {
        "path": "LiheYoung/depth-anything-large-hf",  # Correct HF path for V1
        "vram_mb": 4000,
        "direct_url": "https://github.com/LiheYoung/Depth-Anything/releases/download/v1.0/depth_anything_vitl14.pt",
        "model_type": "v1",
        "encoder": "vitl"
    },
    "Depth-Anything-V2-Small": {
        "path": "depth-anything/Depth-Anything-V2-Small-hf",  # Updated corrected path as shown in example
        "vram_mb": 1500, 
        "direct_url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/pytorch_model.bin",
        "model_type": "v2",
        "encoder": "vits",
        "config": MODEL_CONFIGS["vits"]
    },
    "Depth-Anything-V2-Base": {
        "path": "depth-anything/Depth-Anything-V2-Base-hf",  # Updated corrected path
        "vram_mb": 2500,
        "direct_url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf/resolve/main/pytorch_model.bin",
        "model_type": "v2",
        "encoder": "vitb",
        "config": MODEL_CONFIGS["vitb"]
    },
    # Add MiDaS models as dedicated options with direct download URLs
    "MiDaS-Small": {
        "path": "Intel/dpt-hybrid-midas",
        "vram_mb": 1000,
        "midas_type": "MiDaS_small",
        "direct_url": "https://github.com/intel-isl/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
    },
    "MiDaS-Base": {
        "path": "Intel/dpt-hybrid-midas",
        "vram_mb": 1200,
        "midas_type": "DPT_Hybrid",
        "direct_url": "https://github.com/intel-isl/MiDaS/releases/download/v3/dpt_hybrid-midas-501f0c75.pt"
    },
    # DA3 (Depth Anything V3) Models - Apache 2.0 Licensed (Commercial Friendly)
    # Note: These models require the depth_anything_v3 package to be installed
    "Depth-Anything-V3-Small": {
        "path": "depth-anything/DA3-Small",
        "vram_mb": 2000,  # Estimated: 80M params
        "model_type": "v3",
        "encoder": "vits",
        "license": "Apache-2.0",
        "supports_batch": True,  # Multi-view support
        "supports_pose": True,
        "params": "80M"
    },
    "Depth-Anything-V3-Base": {
        "path": "depth-anything/DA3-Base",
        "vram_mb": 2500,  # Estimated: 120M params
        "model_type": "v3",
        "encoder": "vitb",
        "license": "Apache-2.0",
        "supports_batch": True,  # Multi-view support
        "supports_pose": True,
        "params": "120M"
    },
    "Depth-Anything-V3-Large": {
        "path": "depth-anything/DA3-Large",
        "vram_mb": 4000,
        "model_type": "v3",
        "encoder": "vitl",
        "license": "CC BY-NC 4.0",
        "supports_batch": True,
        "supports_pose": True,
        "params": "350M"
    },
    "Depth-Anything-V3-Giant": {
        "path": "depth-anything/DA3-Giant",
        "vram_mb": 6000,
        "model_type": "v3",
        "encoder": "vitg",
        "license": "CC BY-NC 4.0",
        "supports_batch": True,
        "supports_pose": True,
        "params": "1.15B"
    },
    "Depth-Anything-V3-Nested-Giant-Large": {
        "path": "depth-anything/DA3NESTED-GIANT-LARGE",
        "vram_mb": 7000,
        "model_type": "v3",
        "encoder": "nested",
        "license": "CC BY-NC 4.0",
        "supports_batch": True,
        "supports_pose": True,
        "metric_scaling": True,
        "params": "1.4B"
    },
    "Depth-Anything-V3-Mono-Large": {
        "path": "depth-anything/DA3Mono-Large",
        "vram_mb": 4000,
        "model_type": "v3",
        "encoder": "vitl",
        "license": "Apache-2.0",
        "supports_batch": False,
        "supports_pose": False,
        "params": "350M",
        "note": "Monocular only, no camera estimation"
    },
    "Depth-Anything-V3-Metric-Large": {
        "path": "depth-anything/DA3Metric-Large",
        "vram_mb": 4000,
        "model_type": "v3",
        "encoder": "vitl",
        "license": "Apache-2.0",
        "supports_batch": False,
        "supports_pose": False,
        "metric_depth": True,
        "params": "350M",
        "note": "Metric depth only, no camera estimation"
    },
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
                # Get original dimensions
                original_width, original_height = image.size
                
                # Ensure dimensions are multiple of 32 (required for some models)
                # This helps prevent tensor dimension mismatches
                target_height = ((original_height + 31) // 32) * 32
                target_width = ((original_width + 31) // 32) * 32
                
                # Keep original dimensions - don't force 384x384
                # The caller should already have resized to the requested input_size
                
                # Log resize information if needed
                if (target_width != original_width) or (target_height != original_height):
                    logger.info(f"Adjusting dimensions from {original_width}x{original_height} to {target_width}x{target_height} (multiples of 32)")
                    img_resized = image.resize((target_width, target_height), Image.LANCZOS)
                else:
                    img_resized = image
                
                # Convert to numpy array
                img_np = np.array(img_resized).astype(np.float32) / 255.0
                
                # Check for NaN values and replace them with zeros
                if np.isnan(img_np).any():
                    logger.warning("Input image contains NaN values. Replacing with zeros.")
                    img_np = np.nan_to_num(img_np, nan=0.0)
                
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
                input_tensor = None
                
                # Handle potential error cases with clearer messages
                if not torch.is_tensor(image):
                    logger.error(f"Expected tensor or PIL image, got {type(image)}")
                    # Create dummy tensor as fallback
                    input_tensor = torch.ones((1, 3, 512, 512), dtype=torch.float32)
                elif image.numel() == 0:
                    logger.error("Input tensor is empty")
                    # Create dummy tensor as fallback
                    input_tensor = torch.ones((1, 3, 512, 512), dtype=torch.float32)
                else:
                    # Check for NaN values
                    if torch.isnan(image).any():
                        logger.warning("Input tensor contains NaN values. Replacing with zeros.")
                        image = torch.nan_to_num(image, nan=0.0)
                    
                    # Always convert to float32 to prevent type mismatches
                    input_tensor = image.float()  # Convert any tensor to FloatTensor
                
                # Handle tensor shape issues with more robust dimension checking
                if input_tensor.dim() == 2:  # [H, W]
                    # Single channel 2D tensor
                    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims [1, 1, H, W]
                    logger.info(f"Converted 2D tensor to 4D with shape: {input_tensor.shape}")
                elif input_tensor.dim() == 3:
                    # Could be [C, H, W] or [B, H, W] or [H, W, C]
                    shape = input_tensor.shape
                    if shape[-1] == 3 or shape[-1] == 1:  # [H, W, C] format
                        # Convert from HWC to BCHW
                        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
                        logger.info(f"Converted HWC tensor to BCHW with shape: {input_tensor.shape}")
                    elif shape[0] <= 3:  # Likely [C, H, W]
                        input_tensor = input_tensor.unsqueeze(0)  # Add batch dim [1, C, H, W]
                        logger.info(f"Added batch dimension to CHW tensor: {input_tensor.shape}")
                    else:  # Likely [B, H, W]
                        input_tensor = input_tensor.unsqueeze(1)  # Add channel dim [B, 1, H, W]
                        logger.info(f"Added channel dimension to BHW tensor: {input_tensor.shape}")
                
                # Ensure proper shape after corrections
                if input_tensor.dim() != 4:
                    logger.warning(f"Tensor still has incorrect dimensions ({input_tensor.dim()}). Forcing reshape.")
                    # Force reshape to 4D
                    orig_shape = input_tensor.shape
                    if input_tensor.dim() > 4:
                        # Too many dimensions, collapse extras
                        input_tensor = input_tensor.reshape(1, -1, orig_shape[-2], orig_shape[-1])
                    else:
                        # Create a standard 4D tensor as fallback
                        input_tensor = torch.ones((1, 3, 512, 512), dtype=torch.float32)
            
            # Move to device and ensure float type
            input_tensor = input_tensor.to(self.device).float()
            
            # Log tensor shape for debugging
            logger.info(f"MiDaS input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            
            # Run inference with better error handling
            with torch.no_grad():
                try:
                    # Make sure input is float32 and model weights are float32
                    output = self.model(input_tensor)
                    
                    # Handle various output shapes
                    if output.dim() == 1:  # [B*H*W] flattened output
                        # Reshape based on input dimensions
                        b, _, h, w = input_tensor.shape
                        output = output.reshape(b, 1, h, w)
                    elif output.dim() == 2:  # [B, H*W] or similar
                        # Could be flattened spatial dimensions
                        b = output.shape[0]
                        if b == input_tensor.shape[0]:  # Batch size matches
                            h = int(np.sqrt(output.shape[1]))  # Estimate height assuming square
                            w = h
                            if h * w == output.shape[1]:  # Perfect square
                                output = output.reshape(b, 1, h, w)
                            else:
                                # Not a perfect square, use input dimensions
                                _, _, h, w = input_tensor.shape
                                output = output.reshape(b, 1, h, w)
                        else:
                            # Add dimensions to make 4D
                            output = output.unsqueeze(1).unsqueeze(1)
                    
                    # Ensure output has standard 4D shape (B,C,H,W) for interpolation
                    if output.dim() != 4:
                        logger.warning(f"Output has non-standard dimensions: {output.shape}, adding dimensions")
                        # Add dimensions until we have 4D
                        while output.dim() < 4:
                            output = output.unsqueeze(-1)
                    
                    # Resize to match input resolution
                    if isinstance(image, Image.Image):
                        w, h = image.size
                        
                        # Log the shape for debugging
                        logger.info(f"Resizing output tensor from shape {output.shape} to size ({h}, {w})")
                        
                        # Ensure output tensor has correct number of dimensions for interpolation
                        # Standard interpolation requires 4D tensor (B,C,H,W)
                        try:
                            # Now interpolate with proper dimensions
                            output = torch.nn.functional.interpolate(
                                output,
                                size=(h, w),
                                mode="bicubic",
                                align_corners=False
                            )
                        except RuntimeError as resize_err:
                            logger.error(f"Interpolation error: {resize_err}. Attempting to fix tensor shape.")
                            
                            # Last resort: create compatible tensor from output data
                            try:
                                # Get data and reshape to simple 2D first
                                output_data = output.view(-1).cpu().numpy()
                                output_reshaped = torch.from_numpy(
                                    np.resize(output_data, (h * w))
                                ).reshape(1, 1, h, w).to(self.device).float()
                                
                                logger.info(f"Corrected output shape to {output_reshaped.shape}")
                                output = output_reshaped
                            except Exception as reshape_err:
                                logger.error(f"Reshape fix failed: {reshape_err}. Using fallback tensor.")
                                # Create a basic gradient as fallback
                                output = torch.ones((1, 1, h, w), device=self.device, dtype=torch.float32)
                                y_coords = torch.linspace(0, 1, h).reshape(-1, 1).repeat(1, w)
                                output[0, 0, :, :] = y_coords.to(self.device)
                
                except Exception as model_err:
                    logger.error(f"Model inference error: {model_err}")
                    logger.error(traceback.format_exc())
                    
                    # Create a visually distinguishable gradient pattern fallback
                    if isinstance(image, Image.Image):
                        w, h = image.size
                    else:
                        # Extract dimensions from input tensor
                        _, _, h, w = input_tensor.shape if input_tensor.dim() >= 4 else (1, 1, 512, 512)
                    
                    # Create gradient depth map as fallback
                    output = torch.ones((1, 1, h, w), device=self.device, dtype=torch.float32)
                    y_coords = torch.linspace(0, 1, h).reshape(-1, 1).repeat(1, w)
                    output[0, 0, :, :] = y_coords.to(self.device)
                
            # Final validation - ensure output is float32 and has no NaNs
            output = output.float()
            if torch.isnan(output).any():
                logger.warning("Output contains NaN values. Replacing with zeros.")
                output = torch.nan_to_num(output, nan=0.0)
            
            # Use same interface as the pipeline
            return {"predicted_depth": output}
            
        except Exception as e:
            logger.error(f"Error in MiDaS inference: {e}")
            logger.error(traceback.format_exc())
            
            # Return a placeholder depth map
            if isinstance(image, Image.Image):
                w, h = image.size
                dummy_tensor = torch.ones((1, 1, h, w), device=self.device)
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
                dummy_tensor = torch.ones((1, 1, h, w), device=self.device)
            
            return {"predicted_depth": dummy_tensor}

@dataclass
class DA3Prediction:
    """
    Structured output from DA3 models containing all prediction fields.
    """
    depth: torch.Tensor  # [N, H, W] normalized 0-1
    confidence: Optional[torch.Tensor] = None  # [N, H, W] or None
    extrinsics: Optional[torch.Tensor] = None  # [N, 3, 4] or None
    intrinsics: Optional[torch.Tensor] = None  # [N, 3, 3] or None
    processed_images: Optional[torch.Tensor] = None  # [N, H, W, 3] uint8
    raw_depth: Optional[torch.Tensor] = None  # [N, H, W] unnormalized
    supports_pose: bool = False

class DA3ModelWrapper:
    """
    Enhanced wrapper for Depth Anything V3 API.
    """
    def __init__(self, model, device, model_name: str):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.supports_pose = self._check_pose_support(model_name)
        logger.info(f"DA3ModelWrapper initialized: {model_name}, Pose support: {self.supports_pose}, Device: {device}")

    def _check_pose_support(self, model_name: str) -> bool:
        """Check if model supports camera pose estimation (DA3 variants only)."""
        model_lower = model_name.lower()
        # Mono/metric variants don't support pose estimation
        if "mono" in model_lower or "metric" in model_lower:
            return False
        # DA3 models with size variants support pose estimation
        is_da3 = "v3" in model_lower or "da3" in model_lower
        size_variants = ["small", "base", "large", "giant", "nested"]
        return is_da3 and any(v in model_lower for v in size_variants)

    def __call__(self, image: Union[Image.Image, List[Image.Image]]) -> DA3Prediction:
        try:
            if isinstance(image, Image.Image):
                images = [image]
                is_batch = False
            else:
                images = list(image)
                is_batch = True

            with torch.inference_mode():
                prediction = self.model.inference(images)

            # Extract depth maps
            raw_depths = prediction.depth if hasattr(prediction, 'depth') else np.array(prediction)

            # Normalize depths
            normalized_depths = []
            for depth in raw_depths:
                depth_min, depth_max = depth.min(), depth.max()
                if depth_max - depth_min > 1e-6:
                    norm_depth = (depth - depth_min) / (depth_max - depth_min)
                else:
                    norm_depth = np.zeros_like(depth)
                normalized_depths.append(norm_depth)

            depth_array = np.stack(normalized_depths, axis=0)
            depth_tensor = torch.from_numpy(depth_array).float().to(self.device)
            raw_depth_tensor = torch.from_numpy(raw_depths).float().to(self.device)

            # Extract confidence
            confidence_tensor = None
            if hasattr(prediction, 'conf') and prediction.conf is not None:
                conf_array = prediction.conf
                confidence_tensor = torch.from_numpy(conf_array).float().to(self.device)

            # Extract camera parameters
            extrinsics_tensor = None
            intrinsics_tensor = None
            if self.supports_pose:
                if hasattr(prediction, 'extrinsics') and prediction.extrinsics is not None:
                    ext_array = prediction.extrinsics
                    extrinsics_tensor = torch.from_numpy(ext_array).float().to(self.device)
                if hasattr(prediction, 'intrinsics') and prediction.intrinsics is not None:
                    int_array = prediction.intrinsics
                    intrinsics_tensor = torch.from_numpy(int_array).float().to(self.device)

            # Extract processed images
            processed_tensor = None
            if hasattr(prediction, 'processed_images') and prediction.processed_images is not None:
                proc_array = prediction.processed_images
                processed_tensor = torch.from_numpy(proc_array).to(self.device)

            # Handle single image case - squeeze batch dimension from all tensors
            if not is_batch:
                tensors = [depth_tensor, raw_depth_tensor, confidence_tensor,
                           extrinsics_tensor, intrinsics_tensor, processed_tensor]
                depth_tensor, raw_depth_tensor, confidence_tensor, \
                    extrinsics_tensor, intrinsics_tensor, processed_tensor = \
                    [t.squeeze(0) if t is not None else None for t in tensors]

            return DA3Prediction(
                depth=depth_tensor,
                confidence=confidence_tensor,
                extrinsics=extrinsics_tensor,
                intrinsics=intrinsics_tensor,
                processed_images=processed_tensor,
                raw_depth=raw_depth_tensor,
                supports_pose=self.supports_pose
            )

        except Exception as e:
            logger.error(f"Error in DA3 inference: {e}")
            logger.error(traceback.format_exc())
            if isinstance(image, Image.Image): w, h = image.size
            elif isinstance(image, list) and len(image) > 0: w, h = image[0].size
            else: w, h = 512, 512
            dummy_depth = torch.ones((h, w), device=self.device, dtype=torch.float32)
            return DA3Prediction(depth=dummy_depth, supports_pose=False)

    def eval(self):
        if hasattr(self.model, 'eval'): self.model.eval()
        return self

    def to(self, device):
        if hasattr(self.model, 'to'): self.model = self.model.to(device)
        self.device = device
        return self


def get_available_models():
    """
    Returns a list of available depth models based on installed dependencies.

    DA3 models are only included if depth_anything_v3 package is installed.
    """
    available = []
    for model_name, model_info in DEPTH_MODELS.items():
        model_type = model_info.get("model_type", "v1") if isinstance(model_info, dict) else "v1"

        # DA3 models require the depth_anything_v3 package
        if model_type == "v3" and not DA3_AVAILABLE:
            continue

        available.append(model_name)

    return available


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
        """Define the input types for the node.

        Note: DA3 models are only shown if depth_anything_v3 package is installed.
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (get_available_models(),),
            },
            "optional": {
                "input_size": ("INT", {"default": 1024, "min": 384, "max": 8192, "step": 32}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "median_size": ("INT", {"default": 0, "min": 0, "max": 21, "step": 2}),  # Odd values only, max 21 for PIL
                "apply_auto_contrast": ("BOOLEAN", {"default": False}),
                "apply_gamma": ("BOOLEAN", {"default": False}),
                "force_reload": ("BOOLEAN", {"default": False}),
                "force_cpu": ("BOOLEAN", {"default": False}),
                "enable_camera_estimation": ("BOOLEAN", {"default": True}),
                "output_raw_depth": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        CAMERA_EXTRINSICS,
        CAMERA_INTRINSICS,
        "STRING"
    )
    RETURN_NAMES = (
        "depth",
        "confidence",
        "extrinsics",
        "intrinsics",
        "camera_json"
    )
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
            # Check for valid model name with more helpful fallback
            if model_name not in DEPTH_MODELS:
                # Find the most similar model name if possible
                available_models = list(DEPTH_MODELS.keys())
                
                if len(available_models) > 0:
                    # First try to find a model with a similar name
                    model_name_lower = model_name.lower()
                    
                    # Prioritized fallback selection logic:
                    # 1. Try to find a model with a similar name
                    # 2. Prefer V3 models if V3 was requested (and DA3 is available)
                    # 3. Prefer V2 models if V2 was requested
                    # 4. Prefer smaller models (more reliable)
                    if "v3" in model_name_lower and DA3_AVAILABLE:
                        if "small" in model_name_lower:
                            fallback_model = "Depth-Anything-V3-Small"
                        else:
                            fallback_model = "Depth-Anything-V3-Base"
                    elif "v2" in model_name_lower and "small" in model_name_lower:
                        fallback_model = "Depth-Anything-V2-Small"
                    elif "v2" in model_name_lower and "base" in model_name_lower:
                        fallback_model = "Depth-Anything-V2-Base"
                    elif "v2" in model_name_lower:
                        fallback_model = "Depth-Anything-V2-Small"
                    elif "small" in model_name_lower:
                        fallback_model = "Depth-Anything-Small"
                    elif "midas" in model_name_lower:
                        fallback_model = "MiDaS-Small"
                    else:
                        # Default to the first model if no better match found
                        fallback_model = available_models[0]
                    
                    logger.warning(f"Unknown model: {model_name}. Falling back to {fallback_model}")
                    model_name = fallback_model
                else:
                    raise ValueError(f"No depth models available. Please check your installation.")
            
            # Get model info and validate
            model_info = DEPTH_MODELS[model_name]
            
            # Handle model_info as string or dict with better defaults
            if isinstance(model_info, dict):
                model_path = model_info.get("path", "")
                required_vram = model_info.get("vram_mb", 2000) * 1024  # Convert to KB
                model_type = model_info.get("model_type", "v1")  # v1 or v2 
                encoder = model_info.get("encoder", "vits")  # Model encoder type
                config = model_info.get("config", None)  # Model config for direct loading
                direct_url = model_info.get("direct_url", None)  # Direct download URL
            else:
                model_path = str(model_info)
                required_vram = 2000 * 1024  # Default 2GB
                model_type = "v1"
                encoder = "vits"
                config = None
                direct_url = None
            
            # Only reload if needed or forced
            if not force_reload and self.depth_estimator is not None and self.current_model == model_path:
                logger.info(f"Model '{model_name}' already loaded")
                return
            
            # Clean up any existing model to free memory before loading new one
            self.cleanup()
            
            # Set up device for model
            if self.device is None:
                self.device = get_torch_device()
            
            logger.info(f"Loading depth model: {model_name} on {'CPU' if force_cpu else self.device}")
            
            # Enhanced VRAM check with better error handling
            if torch.cuda.is_available() and not force_cpu:
                try:
                    free_mem_info = get_free_memory(self.device)
                    
                    # Process different return formats from get_free_memory
                    if isinstance(free_mem_info, tuple):
                        free_mem, total_mem = free_mem_info
                        logger.info(f"Available VRAM: {free_mem/1024:.2f}MB, Required: {required_vram/1024:.2f}MB")
                    else:
                        free_mem = free_mem_info
                        logger.info(f"Available VRAM: {free_mem/1024:.2f}MB, Required: {required_vram/1024:.2f}MB")
                        total_mem = free_mem * 2  # Estimate if not available
                    
                    # Add buffer to required memory to avoid OOM issues
                    required_vram_with_buffer = required_vram * 1.2  # Add 20% buffer
                    
                    # If not enough memory, fall back to CPU with warning
                    if free_mem < required_vram_with_buffer:
                        logger.warning(
                            f"Insufficient VRAM for {model_name} (need ~{required_vram/1024:.1f}MB, " + 
                            f"have {free_mem/1024:.1f}MB). Using CPU instead."
                        )
                        force_cpu = True
                except Exception as mem_error:
                    logger.warning(f"Error checking VRAM: {str(mem_error)}. Using CPU to be safe.")
                    force_cpu = True
            
            # Determine optimal device configuration
            device_type = 'cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Use appropriate dtype based on device and model
            # FP16 for CUDA saves VRAM but doesn't work well for all models
            if 'cuda' in str(self.device) and not force_cpu:
                # V2 and V3 models have issues with FP16 - use FP32 for them
                if model_type in ("v2", "v3"):
                    dtype = torch.float32
                else:
                    # Other models can use FP16 to save VRAM
                    dtype = torch.float16
            else:
                # CPU always uses FP32
                dtype = torch.float32

            # Special handling for V3 (DA3) models - use DA3 API instead of transformers pipeline
            if model_type == "v3":
                if not DA3_AVAILABLE:
                    raise RuntimeError(
                        f"DA3 model '{model_name}' requested but depth_anything_v3 package not installed. "
                        "Please install with: pip install depth-anything-v3"
                        f"DA3 model '{model_name}' requested but depth_anything_3 package not installed. "
                        "Please install with: pip install depth-anything-3"
                    )

                logger.info(f"Loading DA3 model: {model_name} using Depth Anything V3 API")
                da3_model = None
                try:
                    # Load DA3 model from HuggingFace
                    da3_model = DepthAnything3.from_pretrained(model_path)

                    # Move to appropriate device
                    target_device = self.device if not force_cpu else 'cpu'
                    da3_model = da3_model.to(target_device)

                    # Set to eval mode
                    da3_model.eval()

                    # Wrap in DA3ModelWrapper for compatibility
                    if da3_model is not None:
                        self.depth_estimator = DA3ModelWrapper(da3_model, self.device if not force_cpu else 'cpu', model_name)
                    else:
                        # Fallback (shouldn't happen if check passed)
                        logger.error("DA3 model loaded as None")
                    self.current_model = model_path
                    logger.info(f"Successfully loaded DA3 model: {model_name}")
                    return

                except Exception as da3_error:
                    logger.error(f"Failed to load DA3 model '{model_name}': {str(da3_error)}")
                    logger.error(traceback.format_exc())

                    # Fall back to V2 if DA3 loading fails
                    logger.warning(f"Falling back to Depth-Anything-V2-Small due to DA3 loading failure")
                    return self.ensure_model_loaded("Depth-Anything-V2-Small", True, force_cpu)
            
            # Create model-specific cache directory
            # Use consistent naming to improve cache hits
            model_cache_name = model_name.replace("-", "_").lower()
            cache_dir = os.path.join(MODELS_DIR, model_cache_name)
            os.makedirs(cache_dir, exist_ok=True)
            
            # Prioritized loading strategy:
            # 1. Check for locally cached model files
            # 2. Try direct download from URLs that don't require authentication
            # 3. Try loading from Hugging Face using transformers pipeline
            # 4. Fall back to direct model implementation
            # 5. Fall back to MiDaS model
            
            # Step 1: First check if we already have a local model file
            local_model_file = None
            
            # Search all valid model directories for existing files
            for base_path in existing_paths:
                # Check multiple possible locations and naming patterns
                locations_to_check = [
                    os.path.join(base_path, model_cache_name),
                    os.path.join(base_path, model_path.replace("/", "_")),
                    base_path,
                    os.path.join(base_path, "v2") if model_type == "v2" else None,
                ]
                
                # Filter out None values
                locations_to_check = [loc for loc in locations_to_check if loc is not None]
                
                # Common model filenames to check
                model_filenames = [
                    "pytorch_model.bin",
                    "model.pt",
                    "model.pth",
                    f"{model_cache_name}.pt",
                    f"{model_cache_name}.bin",
                    f"depth_anything_{encoder}.pt",  # Common naming for Depth Anything models
                    f"depth_anything_v2_{encoder}.pt",  # V2 naming format
                ]
                
                # Search all locations and filenames
                for location in locations_to_check:
                    if os.path.exists(location):
                        for filename in model_filenames:
                            file_path = os.path.join(location, filename)
                            if os.path.exists(file_path) and os.path.getsize(file_path) > 1000000:  # >1MB to avoid empty files
                                local_model_file = file_path
                                logger.info(f"Found existing model file: {local_model_file}")
                                break
                    
                    if local_model_file:
                        break
                
                if local_model_file:
                    break
            
            # Step 2: If no local file found, try downloading from direct URLs
            # These URLs don't require authentication and are more reliable
            if not local_model_file and model_type == "v2":
                # Comprehensive list of URLs to try for V2 models
                alternative_urls = {
                    "Depth-Anything-V2-Small": [
                        "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/pytorch_model.bin",
                        "https://github.com/LiheYoung/Depth-Anything/releases/download/v2.0/depth_anything_v2_small.pt",
                        "https://huggingface.co/ckpt/depth-anything-v2/resolve/main/depth_anything_v2_small.pt"
                    ],
                    "Depth-Anything-V2-Base": [
                        "https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf/resolve/main/pytorch_model.bin", 
                        "https://github.com/LiheYoung/Depth-Anything/releases/download/v2.0/depth_anything_v2_base.pt",
                        "https://huggingface.co/ckpt/depth-anything-v2/resolve/main/depth_anything_v2_base.pt"
                    ],
                    "MiDaS-Small": [
                        "https://github.com/intel-isl/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
                    ],
                    "MiDaS-Base": [
                        "https://github.com/intel-isl/MiDaS/releases/download/v3/dpt_hybrid-midas-501f0c75.pt"
                    ]
                }
                
                # Get URLs to try (including the direct_url from model_info)
                urls_to_try = []
                if direct_url:
                    urls_to_try.append(direct_url)
                
                # Add alternative URLs for this specific model if available
                if model_name in alternative_urls:
                    urls_to_try.extend(alternative_urls[model_name])
                
                # Try downloading the model if not found locally
                if urls_to_try:
                    for url in urls_to_try:
                        try:
                            # Determine output filename and path
                            model_filename = os.path.basename(url)
                            download_path = os.path.join(cache_dir, model_filename)
                            
                            # Check if already downloaded
                            if os.path.exists(download_path) and os.path.getsize(download_path) > 1000000:  # >1MB to avoid empty files
                                logger.info(f"Found existing downloaded model at {download_path}")
                                local_model_file = download_path
                                break
                            
                            # Create parent directory if needed
                            os.makedirs(os.path.dirname(download_path), exist_ok=True)
                            
                            # Download using a reliable method with multiple retries
                            logger.info(f"Downloading model from {url} to {download_path}")
                            
                            success = False
                            
                            # Try wget first (most reliable for large files)
                            try:
                                import wget
                                wget.download(url, out=download_path)
                                if os.path.exists(download_path) and os.path.getsize(download_path) > 1000000:
                                    logger.info(f"Successfully downloaded model with wget to {download_path}")
                                    success = True
                            except Exception as wget_error:
                                logger.warning(f"wget download failed: {str(wget_error)}")
                            
                            # Try requests if wget failed
                            if not success:
                                try:
                                    # Download with progress reporting
                                    with requests.get(url, stream=True, timeout=60) as response:
                                        response.raise_for_status()
                                        total_size = int(response.headers.get('content-length', 0))
                                        
                                        with open(download_path, 'wb') as f:
                                            downloaded = 0
                                            for chunk in response.iter_content(chunk_size=8192):
                                                f.write(chunk)
                                                downloaded += len(chunk)
                                                
                                                if total_size > 0 and downloaded % (20 * 1024 * 1024) == 0:  # Log every 20MB
                                                    percent = int(100 * downloaded / total_size)
                                                    logger.info(f"Download progress: {downloaded/1024/1024:.1f}MB of {total_size/1024/1024:.1f}MB ({percent}%)")
                                    
                                    if os.path.exists(download_path) and os.path.getsize(download_path) > 1000000:
                                        logger.info(f"Successfully downloaded model with requests to {download_path}")
                                        success = True
                                except Exception as req_error:
                                    logger.warning(f"requests download failed: {str(req_error)}")
                            
                            # Try urllib as last resort
                            if not success:
                                try:
                                    import urllib.request
                                    urllib.request.urlretrieve(url, download_path)
                                    if os.path.exists(download_path) and os.path.getsize(download_path) > 1000000:
                                        logger.info(f"Successfully downloaded model with urllib to {download_path}")
                                        success = True
                                except Exception as urllib_error:
                                    logger.warning(f"urllib download failed: {str(urllib_error)}")
                            
                            # Set the local_model_file if download succeeded
                            if success:
                                local_model_file = download_path
                                break
                            else:
                                # Clean up failed download
                                if os.path.exists(download_path):
                                    try:
                                        os.remove(download_path)
                                    except:
                                        pass
                                
                        except Exception as download_error:
                            logger.warning(f"Failed to download from {url}: {str(download_error)}")
                            continue
            
            # Step 3: Try loading with transformers pipeline
            # This is the most feature-complete approach but may fail with auth issues
            logger.info(f"Trying to load model '{model_name}' using transformers pipeline")
            
            # Priority-ordered list of model paths to try
            # Ordered from most to least likely to work
            model_paths_to_try = []
            
            # Start with the specific model requested
            model_paths_to_try.append(model_path)
            
            # Add V2-specific paths for V2 models
            if model_type == "v2":
                if "small" in model_name.lower():
                    model_paths_to_try.append("depth-anything/Depth-Anything-V2-Small-hf")
                elif "base" in model_name.lower():
                    model_paths_to_try.append("depth-anything/Depth-Anything-V2-Base-hf")
                
                # Add variants with and without -hf suffix
                model_paths_to_try.append(model_path.replace("-hf", ""))
                if "-hf" not in model_path:
                    model_paths_to_try.append(model_path + "-hf")
                
                # Try both organization name formats
                model_paths_to_try.append(model_path.replace("LiheYoung", "depth-anything"))
                model_paths_to_try.append(model_path.replace("depth-anything", "LiheYoung"))
            else:
                # For V1 models, add common variants
                model_paths_to_try.append(model_path.replace("-hf", ""))
                if "-hf" not in model_path:
                    model_paths_to_try.append(model_path + "-hf")
            
            # Add MiDaS fallbacks only if not already trying MiDaS
            if "midas" not in model_name.lower():
                model_paths_to_try.append("Intel/dpt-hybrid-midas")
            
            # Remove duplicates while preserving order
            model_paths_to_try = list(dict.fromkeys(model_paths_to_try))
            
            # Log all paths we're going to try
            logger.info(f"Will try loading from these paths in order: {model_paths_to_try}")
            
            # Try loading with transformers pipeline
            pipeline_success = False
            pipeline_error = None
            
            for path in model_paths_to_try:
                # Skip empty paths
                if not path.strip():
                    continue
                    
                logger.info(f"Attempting to load model from: {path}")
                
                # First try online loading (allows downloading new models)
                try:
                    logger.info(f"Loading with online mode: model={path}, device={device_type}, dtype={dtype}")
                    
                    # Try standard pipeline creation first
                    try:
                        from transformers import pipeline
                        
                        # Create pipeline with timeout and error handling
                        self.depth_estimator = pipeline(
                            "depth-estimation",
                            model=path,
                            cache_dir=cache_dir,
                            local_files_only=False,  # Try online first
                            device_map=device_type,
                            torch_dtype=dtype
                        )
                        
                        # Verify that pipeline was created
                        if self.depth_estimator is None:
                            raise RuntimeError(f"Pipeline initialization returned None for {path}")
                        
                        # Validate by running a test inference
                        test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
                        try:
                            test_result = self.depth_estimator(test_img)
                            
                            # Further verify the output format
                            if not isinstance(test_result, dict) or "predicted_depth" not in test_result:
                                raise RuntimeError("Invalid output format from pipeline")
                            
                            # Success - log and break
                            logger.info(f"Successfully loaded model from {path} with online mode")
                            pipeline_success = True
                            break
                            
                        except Exception as test_error:
                            logger.warning(f"Pipeline created but test failed: {str(test_error)}")
                            raise
                            
                    except TypeError as type_error:
                        # Handle unpacking errors which are common with older transformers versions
                        logger.warning(f"TypeError when creating pipeline: {str(type_error)}")
                        
                        # Try alternative approach with manual component loading
                        logger.info("Trying manual component loading as alternative...")
                        
                        try:
                            from transformers import AutoModelForDepthEstimation, AutoImageProcessor
                            
                            # Load components separately
                            processor = AutoImageProcessor.from_pretrained(
                                path, cache_dir=cache_dir, local_files_only=False
                            )
                            
                            model = AutoModelForDepthEstimation.from_pretrained(
                                path, cache_dir=cache_dir, local_files_only=False,
                                torch_dtype=dtype
                            )
                            
                            # Move model to correct device
                            if not force_cpu and 'cuda' in device_type:
                                model = model.to(self.device)
                            
                            # Create custom pipeline class
                            class CustomDepthEstimator:
                                def __init__(self, model, processor, device):
                                    self.model = model
                                    self.processor = processor
                                    self.device = device
                                    
                                def __call__(self, image):
                                    # Process image and run model
                                    inputs = self.processor(images=image, return_tensors="pt")
                                    
                                    # Move inputs to correct device
                                    if torch.cuda.is_available() and not force_cpu:
                                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                                    
                                    # Run model
                                    with torch.no_grad():
                                        outputs = self.model(**inputs)
                                    
                                    # Return results in standard format
                                    return {"predicted_depth": outputs.predicted_depth}
                            
                            # Create custom pipeline
                            self.depth_estimator = CustomDepthEstimator(model, processor, self.device)
                            
                            # Test the pipeline
                            test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
                            test_result = self.depth_estimator(test_img)
                            
                            # Verify output format
                            if not isinstance(test_result, dict) or "predicted_depth" not in test_result:
                                raise RuntimeError("Invalid output format from custom pipeline")
                            
                            logger.info(f"Successfully loaded model from {path} with custom pipeline")
                            pipeline_success = True
                            break
                            
                        except Exception as custom_error:
                            logger.warning(f"Custom pipeline creation failed: {str(custom_error)}")
                            raise
                        
                except Exception as online_error:
                    logger.warning(f"Online loading failed for {path}: {str(online_error)}")
                    pipeline_error = online_error
                    
                    # Try local-only mode if online fails (faster and often works with cached files)
                    try:
                        logger.info(f"Trying local-only mode for {path}")
                        
                        from transformers import pipeline
                        
                        self.depth_estimator = pipeline(
                            "depth-estimation",
                            model=path,
                            cache_dir=cache_dir,
                            local_files_only=True,  # Only use local files
                            device_map=device_type,
                            torch_dtype=dtype
                        )
                        
                        # Verify pipeline
                        if self.depth_estimator is None:
                            raise RuntimeError(f"Local pipeline initialization returned None for {path}")
                        
                        # Test with small image
                        test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
                        test_result = self.depth_estimator(test_img)
                        
                        logger.info(f"Successfully loaded model from {path} with local-only mode")
                        pipeline_success = True
                        break
                        
                    except Exception as local_error:
                        logger.warning(f"Local-only mode failed for {path}: {str(local_error)}")
                        pipeline_error = local_error
                        # Continue to next path
            
            # Step 4: If transformers pipeline failed, try direct model loading
            if not pipeline_success:
                logger.info("Pipeline loading failed. Trying direct model implementation.")
                
                # If we have a local model file from previous steps, use it with direct loading
                direct_success = False
                
                if local_model_file:
                    logger.info(f"Loading model directly from: {local_model_file}")
                    
                    # For V2 models, use the DepthAnythingV2 implementation
                    if model_type == "v2" and TIMM_AVAILABLE:
                        try:
                            logger.info(f"Using DepthAnythingV2 implementation with config: {config}")
                            
                            # Create model instance with appropriate config
                            if config:
                                model_instance = DepthAnythingV2(**config)
                            else:
                                # Use default config for this encoder type
                                default_config = MODEL_CONFIGS.get(encoder, MODEL_CONFIGS["vits"])
                                model_instance = DepthAnythingV2(**default_config)
                            
                            # Load state dict from file
                            logger.info(f"Loading weights from {local_model_file}")
                            state_dict = torch.load(local_model_file, map_location="cpu")
                            
                            # Convert state dict to float32
                            if any(v.dtype == torch.float64 for v in state_dict.values() if hasattr(v, 'dtype')):
                                logger.info("Converting state dict from float64 to float32")
                                state_dict = {k: v.float() if hasattr(v, 'dtype') else v for k, v in state_dict.items()}
                            
                            # Try loading with different state dict formats
                            try:
                                if "model" in state_dict:
                                    model_instance.load_state_dict(state_dict["model"])
                                else:
                                    model_instance.load_state_dict(state_dict)
                            except Exception as e:
                                logger.warning(f"Strict loading failed: {str(e)}. Trying non-strict loading.")
                                if "model" in state_dict:
                                    model_instance.load_state_dict(state_dict["model"], strict=False)
                                else:
                                    model_instance.load_state_dict(state_dict, strict=False)
                            
                            # Move to correct device and set eval mode
                            model_instance = model_instance.to(self.device).float().eval()
                            
                            # Test the model
                            test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
                            _ = model_instance(test_img)
                            
                            # Success - assign model
                            self.depth_estimator = model_instance
                            direct_success = True
                            logger.info("Successfully loaded model with direct implementation")
                            
                        except Exception as v2_error:
                            logger.warning(f"DepthAnythingV2 loading failed: {str(v2_error)}")
                    
                    # If V2 direct loading failed or isn't applicable, try MiDaS wrapper 
                    if not direct_success:
                        try:
                            logger.info("Falling back to MiDaS wrapper implementation")
                            
                            # Determine MiDaS model type based on model name
                            midas_type = "DPT_Hybrid"  # Default
                            
                            if "small" in model_name.lower():
                                midas_type = "MiDaS_small"
                            elif "large" in model_name.lower():
                                midas_type = "DPT_Large"
                            
                            # Create MiDaS wrapper with model type
                            midas_model = MiDaSWrapper(midas_type, self.device)
                            
                            # Test the model
                            test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
                            _ = midas_model(test_img)
                            
                            # Success - assign model
                            self.depth_estimator = midas_model
                            direct_success = True
                            logger.info("Successfully loaded MiDaS fallback model")
                            
                        except Exception as midas_error:
                            logger.warning(f"MiDaS wrapper loading failed: {str(midas_error)}")
                
                # Step 5: If all previous attempts failed, try one last MiDaS fallback
                if not direct_success and not pipeline_success:
                    try:
                        logger.info("All model loading attempts failed. Trying basic MiDaS fallback.")
                        
                        # Create simple MiDaS wrapper with default settings
                        midas_model = MiDaSWrapper("dpt_hybrid", self.device)
                        
                        # Test with simple image
                        test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
                        _ = midas_model(test_img)
                        
                        # Assign model
                        self.depth_estimator = midas_model
                        logger.info("Successfully loaded basic MiDaS fallback model")
                        
                    except Exception as final_error:
                        # If we get here, all attempts have failed
                        error_msg = f"All model loading attempts failed for {model_name}. Last error: {str(final_error)}"
                        logger.error(error_msg)
                        
                        # Create a helpful error message with instructions
                        all_model_dirs = "\n".join(existing_paths)
                        
                        # Determine error type for better help message
                        if pipeline_error:
                            error_str = str(pipeline_error).lower()
                            
                            if "unauthorized" in error_str or "401" in error_str or "authentication" in error_str:
                                # Authentication error - guide for manual download
                                error_solution = f"""
AUTHENTICATION ERROR: The model couldn't be downloaded due to Hugging Face authentication requirements.

SOLUTION: 
1. Use force_cpu=True in the node settings
2. Try a different model like MiDaS-Small
3. Download the model manually using one of these direct links:
   - https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/pytorch_model.bin
   - https://github.com/LiheYoung/Depth-Anything/releases/download/v2.0/depth_anything_v2_small.pt
   - https://huggingface.co/ckpt/depth-anything-v2/resolve/main/depth_anything_v2_small.pt

   Save the file to one of these directories:
   {all_model_dirs}
"""
                            elif "cuda" in error_str or "gpu" in error_str or "vram" in error_str or "memory" in error_str:
                                # GPU/memory error
                                error_solution = """
GPU ERROR: The model failed to load on your GPU.

SOLUTION:
1. Use force_cpu=True to use CPU processing instead
2. Reduce input_size parameter to 384 to reduce memory requirements
3. Try a smaller model like MiDaS-Small
4. Ensure you have the latest GPU drivers installed
"""
                            else:
                                # Generic error
                                error_solution = f"""
Failed to load any depth estimation model.

SOLUTION:
1. Use force_cpu=True to use CPU for processing
2. Try using a different model like MiDaS-Small
3. Download a model file manually and place it in one of these directories:
   {all_model_dirs}
4. Restart ComfyUI to ensure clean state
"""
                        else:
                            # Generic error when pipeline_error isn't set
                            error_solution = f"""
Failed to load any depth estimation model.

SOLUTION:
1. Use force_cpu=True to use CPU for processing
2. Try using a different model like MiDaS-Small
3. Download a model file manually and place it in one of these directories:
   {all_model_dirs}
4. Restart ComfyUI to ensure clean state
"""
                        
                        # Raise helpful error
                        raise RuntimeError(f"MODEL LOADING ERROR: {error_solution}")
            
            # Ensure the model is on the correct device
            if hasattr(self.depth_estimator, 'model') and hasattr(self.depth_estimator.model, 'to'):
                if force_cpu:
                    self.depth_estimator.model = self.depth_estimator.model.to('cpu')
                else:
                    self.depth_estimator.model = self.depth_estimator.model.to(self.device)
            
            # Set model to eval mode if applicable
            if hasattr(self.depth_estimator, 'eval'):
                self.depth_estimator.eval()
            elif hasattr(self.depth_estimator, 'model') and hasattr(self.depth_estimator.model, 'eval'):
                self.depth_estimator.model.eval()
            
            # Store current model info
            self.current_model = model_path
            logger.info(f"Model '{model_name}' loaded successfully")
            
        except Exception as e:
            # Clean up on failure
            self.cleanup()
            
            # Log detailed error info
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Re-raise with clear message
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
            
            # Look in all possible model directories
            # This is important to support various directory structures
            model_found = False
            model_path_local = None
            
            # Make a unique model cache directory for this specific model
            model_subfolder = model_name.replace("-", "_").lower()
            
            # Check all possible locations for the model file
            for base_path in existing_paths:
                # Try different possible locations and filename patterns
                possible_model_locations = [
                    # Direct downloads in the model directory
                    os.path.join(base_path, model_subfolder),
                    
                    # Using the full HF directory structure
                    os.path.join(base_path, model_info.get("path", "").replace("/", "_")),
                    
                    # Directly in base directory
                    base_path,
                ]
                
                # Add directory structure with model configs if V2
                if model_info.get("model_type") == "v2":
                    v2_path = os.path.join(base_path, "v2")
                    possible_model_locations.append(v2_path)
                    possible_model_locations.append(os.path.join(v2_path, model_subfolder))
                
                # Try all locations
                logger.info(f"Searching for existing model in these directories: {possible_model_locations}")
                
                for location in possible_model_locations:
                    # Check for model file with various naming patterns
                    if os.path.exists(location):
                        # Check for common filenames
                        for filename in ["pytorch_model.bin", "model.pt", "model.pth", 
                                         f"{model_subfolder}.pt", f"{model_subfolder}.bin"]:
                            file_path = os.path.join(location, filename)
                            if os.path.exists(file_path):
                                logger.info(f"Found existing model file: {file_path}")
                                model_path_local = file_path
                                model_found = True
                                break
                    
                    if model_found:
                        break
                
                if model_found:
                    break
            
            # If model not found, use the first directory for downloading
            cache_dir = os.path.join(existing_paths[0], model_subfolder)
            os.makedirs(cache_dir, exist_ok=True)
            
            # Get model configuration
            model_type = model_info.get("model_type", "v1")
            encoder = model_info.get("encoder", "vits")
            config = model_info.get("config", MODEL_CONFIGS.get(encoder, MODEL_CONFIGS["vits"]))
            
            # Step 1: If model not found locally, download it
            # List of alternative URLs that don't require authentication
            alternative_urls = {
                "Depth-Anything-V2-Small": [
                    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/pytorch_model.bin",
                    "https://github.com/LiheYoung/Depth-Anything/releases/download/v2.0/depth_anything_v2_small.pt",
                    "https://huggingface.co/ckpt/depth-anything-v2/resolve/main/depth_anything_v2_small.pt"
                ],
                "Depth-Anything-V2-Base": [
                    "https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf/resolve/main/pytorch_model.bin",
                    "https://github.com/LiheYoung/Depth-Anything/releases/download/v2.0/depth_anything_v2_base.pt",
                    "https://huggingface.co/ckpt/depth-anything-v2/resolve/main/depth_anything_v2_base.pt"
                ],
                "MiDaS-Base": [
                    "https://github.com/intel-isl/MiDaS/releases/download/v3/dpt_hybrid-midas-501f0c75.pt"
                ]
            }
            
            # Get primary URL from model_info
            direct_url = model_info.get("direct_url")
            
            # Add alternative URLs to try if the main one fails
            urls_to_try = [direct_url] if direct_url else []
            
            # Add alternative URLs for this model if available
            if model_name in alternative_urls:
                urls_to_try.extend(alternative_urls[model_name])
            
            # Try downloading the model if not found locally
            if not model_found and urls_to_try:
                # Try each URL in sequence until one works
                for url in urls_to_try:
                    if not url:
                        continue
                        
                    try:
                        model_filename = os.path.basename(url)
                        model_path_local = os.path.join(cache_dir, model_filename)
                        
                        if os.path.exists(model_path_local):
                            logger.info(f"Model already exists at {model_path_local}")
                            model_found = True
                            break
                        
                        logger.info(f"Attempting to download model from {url} to {model_path_local}")
                        
                        # Create parent directory if needed
                        os.makedirs(os.path.dirname(model_path_local), exist_ok=True)
                        
                        # Try different download methods
                        download_success = False
                        
                        # First try wget (more reliable for large files)
                        try:
                            logger.info(f"Downloading with wget: {url}")
                            wget.download(url, out=model_path_local)
                            logger.info(f"Downloaded model weights to {model_path_local}")
                            download_success = True
                        except Exception as wget_error:
                            logger.warning(f"wget download failed: {str(wget_error)}")
                            
                            # Fallback to requests
                            try:
                                logger.info(f"Downloading with requests: {url}")
                                response = requests.get(url, stream=True)
                                
                                if response.status_code == 200:
                                    total_size = int(response.headers.get('content-length', 0))
                                    logger.info(f"File size: {total_size/1024/1024:.1f} MB")
                                    
                                    with open(model_path_local, 'wb') as f:
                                        downloaded = 0
                                        for data in response.iter_content(1024 * 1024):  # 1MB chunks
                                            f.write(data)
                                            downloaded += len(data)
                                            if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                                                progress = (downloaded / total_size) * 100
                                                logger.info(f"Downloaded {downloaded/1024/1024:.1f}MB of {total_size/1024/1024:.1f}MB ({progress:.1f}%)")
                                    
                                    logger.info(f"Download complete: {model_path_local}")
                                    download_success = True
                                else:
                                    logger.warning(f"Failed to download from {url}: HTTP status {response.status_code}")
                            except Exception as req_error:
                                logger.warning(f"Requests download failed: {str(req_error)}")
                        
                        # Try urllib as last resort
                        if not download_success:
                            try:
                                logger.info(f"Downloading with urllib: {url}")
                                urllib.request.urlretrieve(url, model_path_local)
                                logger.info(f"Downloaded model weights to {model_path_local}")
                                download_success = True
                            except Exception as urllib_error:
                                logger.warning(f"urllib download failed: {str(urllib_error)}")
                        
                        # Check if download succeeded
                        if download_success and os.path.exists(model_path_local) and os.path.getsize(model_path_local) > 0:
                            logger.info(f"Successfully downloaded model to {model_path_local}")
                            model_found = True
                            break
                        else:
                            logger.warning(f"Download appeared to succeed but file is empty or missing")
                            # Try to remove the failed download
                            if os.path.exists(model_path_local):
                                try:
                                    os.remove(model_path_local)
                                except:
                                    pass
                    
                    except Exception as dl_error:
                        logger.warning(f"Error downloading from {url}: {str(dl_error)}")
                        continue
                
                if not model_found:
                    logger.error("All download attempts failed")
            
            # Step 2: Create and load the appropriate model if found
            if model_found and model_path_local and os.path.exists(model_path_local):
                logger.info(f"Found model file at: {model_path_local}")
                
                # Handle V2 models with DepthAnythingV2 implementation
                if model_type == "v2" and TIMM_AVAILABLE:
                    try:
                        logger.info(f"Loading as DepthAnythingV2 model with config: {config}")
                        
                        # Create model with the appropriate configuration
                        model = DepthAnythingV2(**config)
                        
                        # Load weights from checkpoint
                        logger.info(f"Loading weights from {model_path_local}")
                        state_dict = torch.load(model_path_local, map_location=device)
                        
                        # Convert state dict to float32 if needed
                        if any(v.dtype == torch.float64 for v in state_dict.values() if hasattr(v, 'dtype')):
                            logger.info("Converting state dict from float64 to float32")
                            state_dict = {k: v.float() if hasattr(v, 'dtype') else v for k, v in state_dict.items()}
                        
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
                        
                        # Move model to the correct device and ensure float32
                        model = model.float().to(device)
                        model.device = device
                        model.eval()
                        
                        # Test the model
                        logger.info("Testing model with sample image")
                        test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
                        
                        try:
                            _ = model(test_img)
                            logger.info("DepthAnythingV2 model loaded and tested successfully")
                            return model
                        except Exception as test_error:
                            logger.error(f"Error during model test: {str(test_error)}")
                            logger.debug(traceback.format_exc())
                    except Exception as e:
                        logger.error(f"Error loading DepthAnythingV2: {str(e)}")
                        logger.debug(traceback.format_exc())
            
            # Fallback to MiDaS model if V2 loading failed or for V1 models
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
            # Log input information for debugging
            if torch.is_tensor(image):
                logger.info(f"Processing tensor image with shape {image.shape}, dtype {image.dtype}")
            elif isinstance(image, np.ndarray):
                logger.info(f"Processing numpy image with shape {image.shape}, dtype {image.dtype}")
            else:
                logger.warning(f"Unexpected image type: {type(image)}")
            
            # Validate and normalize input_size with improved bounds checking
            if not isinstance(input_size, (int, float)):
                logger.warning(f"Invalid input_size type: {type(input_size)}. Using default 518.")
                input_size = 518
            else:
                try:
                    # Convert to int and constrain to valid range
                    input_size = int(input_size)
                except:
                    logger.warning(f"Error converting input_size to int. Using default 518.")
                    input_size = 518
            
            # Ensure input_size is within valid range
            if input_size < 256:
                logger.warning(f"Input size {input_size} is too small, using 256 instead")
                input_size = 256
            elif input_size > 1024:
                logger.warning(f"Input size {input_size} is too large, using 1024 instead")
                input_size = 1024
            
            # Process tensor input with comprehensive error handling
            if torch.is_tensor(image):
                try:
                    # Check tensor dtype and convert to float32 if needed
                    if image.dtype != torch.float32:
                        logger.info(f"Converting input tensor from {image.dtype} to torch.float32")
                        image = image.float()  # Convert to FloatTensor for consistency
                    
                    # Check for NaN/Inf values and fix them
                    nan_count = torch.isnan(image).sum().item()
                    inf_count = torch.isinf(image).sum().item()
                    
                    if nan_count > 0 or inf_count > 0:
                        logger.warning(f"Input tensor contains {nan_count} NaN and {inf_count} Inf values. Replacing with valid values.")
                        image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # Handle empty or invalid tensors
                    if image.numel() == 0:
                        logger.error("Input tensor is empty (zero elements)")
                        return Image.new('RGB', (512, 512), (128, 128, 128))
                    
                    # Handle tensor with incorrect number of dimensions
                    # We need a 3D or 4D tensor to properly extract the image
                    if image.dim() < 3:
                        logger.warning(f"Input tensor has too few dimensions: {image.dim()}D. Adding dimensions.")
                        # Add dimensions until we have at least 3D tensor
                        while image.dim() < 3:
                            image = image.unsqueeze(0)
                        logger.info(f"Adjusted tensor shape to: {image.shape}")
                    
                    # Normalize values to [0, 1] range if needed
                    if image.max() > 1.0 + 1e-5:  # Allow small floating point error
                        min_val, max_val = image.min().item(), image.max().item()
                        logger.info(f"Input tensor values outside [0,1] range: min={min_val}, max={max_val}. Normalizing.")
                        
                        # Common ranges and conversions
                        if min_val >= 0 and max_val <= 255:
                            # Assume [0-255] range for images
                            image = image / 255.0
                        else:
                            # General normalization
                            image = (image - min_val) / (max_val - min_val)
                    
                    # Extract first image from batch if we have a batch dimension
                    if image.dim() == 4:  # [B, C, H, W]
                        # Use first image in batch
                        image_for_conversion = image[0]
                    else:  # 3D tensor, assumed to be [C, H, W]
                        image_for_conversion = image
                    
                    # Move to CPU for numpy conversion
                    image_for_conversion = image_for_conversion.cpu()
                    
                    # Convert to numpy, handling different layouts
                    if image_for_conversion.shape[0] <= 3:  # [C, H, W] format with 1-3 channels
                        # Standard CHW layout - convert to HWC for PIL
                        image_np = image_for_conversion.permute(1, 2, 0).numpy()
                        image_np = image_np * 255.0  # Scale to [0, 255]
                    else:
                        # Unusual channel count - probably not CHW format
                        logger.warning(f"Unusual channel count for CHW format: {image_for_conversion.shape[0]}. Using reshape logic.")
                        
                        # Try to infer the format and convert appropriately
                        if image_for_conversion.dim() == 3 and image_for_conversion.shape[-1] <= 3:
                            # Likely [H, W, C] format, no need to permute
                            image_np = image_for_conversion.numpy() * 255.0
                        else:
                            # Unknown format - try to reshape intelligently
                            logger.warning("Unable to determine tensor layout. Using first 3 channels.")
                            
                            # Default: take first 3 channels (or fewer if < 3 channels)
                            channels = min(3, image_for_conversion.shape[0])
                            image_np = image_for_conversion[:channels].permute(1, 2, 0).numpy() * 255.0
                    
                    # Ensure we have proper RGB image (3 channels)
                    if len(image_np.shape) == 2:  # Single channel image
                        image_np = np.stack([image_np] * 3, axis=-1)
                    elif image_np.shape[-1] == 1:  # Single channel in last dimension
                        image_np = np.concatenate([image_np] * 3, axis=-1)
                    elif image_np.shape[-1] == 4:  # RGBA image - drop alpha channel
                        image_np = image_np[..., :3]
                    elif image_np.shape[-1] > 4:  # More than 4 channels - use first 3
                        logger.warning(f"Image has {image_np.shape[-1]} channels. Using first 3 channels.")
                        image_np = image_np[..., :3]
                    
                    # Ensure proper data type and range
                    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                    
                except Exception as tensor_error:
                    logger.error(f"Error processing tensor image: {str(tensor_error)}")
                    logger.error(traceback.format_exc())
                    # Create a fallback RGB gradient as placeholder
                    placeholder = np.zeros((512, 512, 3), dtype=np.uint8)
                    # Add a gradient pattern for visual distinction
                    for i in range(512):
                        v = int(i / 512 * 255)
                        placeholder[i, :, 0] = v  # R channel gradient
                        placeholder[:, i, 1] = v  # G channel gradient
                    return Image.fromarray(placeholder)
            
            # Process numpy array input with comprehensive error handling
            elif isinstance(image, np.ndarray):
                try:
                    # Check for NaN/Inf values in array
                    if np.isnan(image).any() or np.isinf(image).any():
                        logger.warning("Input array contains NaN or Inf values. Replacing with zeros.")
                        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # Handle empty or invalid arrays
                    if image.size == 0:
                        logger.error("Input array is empty (zero elements)")
                        return Image.new('RGB', (512, 512), (128, 128, 128))
                    
                    # Convert high-precision types to float32 for consistency
                    if image.dtype == np.float64 or image.dtype == np.float16:
                        logger.info(f"Converting numpy array from {image.dtype} to float32")
                        image = image.astype(np.float32)
                    
                    # Normalize values to [0-1] range if float array
                    if np.issubdtype(image.dtype, np.floating):
                        # Check current range
                        min_val, max_val = image.min(), image.max()
                        
                        # Normalize if outside [0,1] range
                        if min_val < 0.0 - 1e-5 or max_val > 1.0 + 1e-5:  # Allow small floating point error
                            logger.info(f"Normalizing array from range [{min_val:.2f}, {max_val:.2f}] to [0, 1]")
                            
                            # Common ranges and conversions
                            if min_val >= 0 and max_val <= 255:
                                # Assume [0-255] float range
                                image = image / 255.0
                            else:
                                # General min-max normalization
                                image = (image - min_val) / (max_val - min_val)
                        
                        # Convert normalized float to uint8 for PIL
                        image_np = (image * 255).astype(np.uint8)
                    elif np.issubdtype(image.dtype, np.integer):
                        # For integer types, check if normalization is needed
                        max_val = image.max()
                        if max_val > 255:
                            logger.info(f"Scaling integer array with max value {max_val} to 0-255 range")
                            # Scale the array to 0-255 range
                            scaled = (image.astype(np.float32) / max_val) * 255
                            image_np = scaled.astype(np.uint8)
                        else:
                            # Already in valid range
                            image_np = image.astype(np.uint8)
                    else:
                        # Unsupported dtype - convert through float32
                        logger.warning(f"Unsupported dtype: {image.dtype}. Converting through float32.")
                        image_np = (image.astype(np.float32) * 255).astype(np.uint8)
                    
                    # Handle different channel configurations and dimensions
                    if len(image_np.shape) == 2:
                        # Grayscale image - convert to RGB
                        logger.info("Converting grayscale image to RGB")
                        image_np = np.stack([image_np] * 3, axis=-1)
                    elif len(image_np.shape) == 3:
                        # Check channel dimension
                        if image_np.shape[-1] == 1:
                            # Single-channel 3D array - convert to RGB
                            image_np = np.concatenate([image_np] * 3, axis=-1)
                        elif image_np.shape[-1] == 4:
                            # RGBA image - drop alpha channel
                            image_np = image_np[..., :3]
                        elif image_np.shape[-1] > 4:
                            # More than 4 channels - use first 3
                            logger.warning(f"Image has {image_np.shape[-1]} channels. Using first 3 channels.")
                            image_np = image_np[..., :3]
                        elif image_np.shape[-1] < 3:
                            # Less than 3 channels but not 1 - unusual case
                            logger.warning(f"Unusual channel count: {image_np.shape[-1]}. Expanding to RGB.")
                            # Repeat the channels to get 3
                            channels = [image_np[..., i % image_np.shape[-1]] for i in range(3)]
                            image_np = np.stack(channels, axis=-1)
                    elif len(image_np.shape) > 3:
                        # More than 3 dimensions - attempt to extract a valid image
                        logger.warning(f"Array has {len(image_np.shape)} dimensions. Attempting to extract 3D slice.")
                        
                        # Try to get a 3D slice with channels in the last dimension
                        if image_np.shape[-1] <= 3:
                            # Extract first instance of higher dimensions
                            while len(image_np.shape) > 3:
                                image_np = image_np[0]
                            
                            # If we have less than 3 channels, expand to RGB
                            if image_np.shape[-1] < 3:
                                channels = [image_np[..., i % image_np.shape[-1]] for i in range(3)]
                                image_np = np.stack(channels, axis=-1)
                        else:
                            # Channels not in last dimension - reshape based on assumptions
                            logger.warning("Could not determine valid layout. Creating placeholder.")
                            image_np = np.zeros((512, 512, 3), dtype=np.uint8)
                            # Add gradient for visual distinction
                            for i in range(512):
                                v = int(i / 512 * 255)
                                image_np[i, :, 0] = v
                    
                except Exception as numpy_error:
                    logger.error(f"Error processing numpy array: {str(numpy_error)}")
                    logger.error(traceback.format_exc())
                    # Create a fallback pattern as placeholder
                    placeholder = np.zeros((512, 512, 3), dtype=np.uint8)
                    # Add checkboard pattern
                    for i in range(0, 512, 32):
                        for j in range(0, 512, 32):
                            if (i//32 + j//32) % 2 == 0:
                                placeholder[i:i+32, j:j+32] = 200
                    return Image.fromarray(placeholder)
            
            # Fallback for non-tensor, non-numpy inputs
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return Image.new('RGB', (512, 512), (100, 100, 150))  # Distinct color for type errors
            
            # Convert to PIL image with error handling
            try:
                pil_image = Image.fromarray(image_np)
            except Exception as pil_error:
                logger.error(f"Error creating PIL image: {str(pil_error)}")
                # Try shape correction if possible
                try:
                    if len(image_np.shape) != 3 or image_np.shape[-1] not in [1, 3, 4]:
                        logger.warning(f"Invalid array shape for PIL: {image_np.shape}")
                        # Create valid RGB array as fallback
                        image_np = np.zeros((512, 512, 3), dtype=np.uint8)
                        pil_image = Image.fromarray(image_np)
                    else:
                        # Other error - use placeholder
                        pil_image = Image.new('RGB', (512, 512), (128, 128, 128))
                except:
                    # Ultimate fallback
                    pil_image = Image.new('RGB', (512, 512), (128, 128, 128))
            
            # Resize the image while preserving aspect ratio and ensuring multiple of 32
            # The multiple of 32 constraint helps prevent tensor dimension errors
            width, height = pil_image.size
            logger.info(f"Original PIL image size: {width}x{height}")
            
            # Determine which dimension to scale to input_size
            if width > height:
                new_width = input_size
                new_height = int(height * (new_width / width))
            else:
                new_height = input_size
                new_width = int(width * (new_height / height))
            
            # Ensure dimensions are multiples of 32 for better compatibility
            new_width = ((new_width + 31) // 32) * 32
            new_height = ((new_height + 31) // 32) * 32
            
            # Resize the image with antialiasing
            try:
                resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
                
                # Verify the resized image
                if resized_image.size[0] <= 0 or resized_image.size[1] <= 0:
                    raise ValueError(f"Invalid resize dimensions: {resized_image.size}")
                
                return resized_image
            except Exception as resize_error:
                logger.error(f"Error during image resize: {str(resize_error)}")
                
                # Try a simpler resize method as fallback
                try:
                    logger.info("Trying fallback resize method")
                    resized_image = pil_image.resize((new_width, new_height), Image.NEAREST)
                    return resized_image
                except:
                    # Last resort - return original image or placeholder
                    if width > 0 and height > 0:
                        logger.warning("Fallback resize failed. Returning original image.")
                        return pil_image
                    else:
                        logger.warning("Invalid original image. Returning placeholder.")
                        return Image.new('RGB', (512, 512), (128, 128, 128))
            
        except Exception as e:
            # Global catch-all handler
            logger.error(f"Error processing image: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Create a visually distinct placeholder image
            placeholder = Image.new('RGB', (512, 512), (120, 80, 80))
            
            try:
                # Add error text to the image for better user feedback
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(placeholder)
                
                # Try to get a font, fall back to default if needed
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                # Add error text
                error_text = str(e)
                # Limit error text length
                if len(error_text) > 60:
                    error_text = error_text[:57] + "..."
                
                # Draw error message
                draw.text((10, 10), "Image Processing Error", fill=(255, 50, 50), font=font)
                draw.text((10, 40), error_text, fill=(255, 255, 255), font=font)
            except:
                # Error adding text - just return the plain placeholder
                pass
            
            return placeholder

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

    def _serialize_camera_data(
        self,
        extrinsics: Optional[torch.Tensor],
        intrinsics: Optional[torch.Tensor],
        confidence: Optional[torch.Tensor],
        raw_depth: Optional[torch.Tensor],
        model_name: str,
        image_width: int,
        image_height: int
    ) -> str:
        """
        Serialize camera data to comprehensive JSON string.

        Args:
            extrinsics: [N, 3, 4] camera extrinsics (opencv w2c format)
            intrinsics: [N, 3, 3] camera intrinsics
            confidence: [N, H, W] confidence maps (optional)
            raw_depth: [N, H, W] raw depth values (optional)
            model_name: Name of the model used
            image_width: Original image width
            image_height: Original image height

        Returns:
            JSON string with camera parameters and metadata
        """
        import json

        data = {
            "format_version": "1.0",
            "model": model_name,
            "timestamp": time.time(),
            "image_width": image_width,
            "image_height": image_height,
        }

        # Serialize extrinsics
        if extrinsics is not None:
            ext_np = extrinsics.cpu().numpy()
            num_views = ext_np.shape[0] if ext_np.ndim == 3 else 1
            data["num_views"] = num_views

            if ext_np.ndim == 3:  # Batch [N, 3, 4]
                data["extrinsics"] = [
                    {
                        "matrix": ext.tolist(),
                        "format": "opencv_w2c",
                        "rotation": ext[:3, :3].tolist(),
                        "translation": ext[:3, 3].tolist() if ext.shape[1] > 3 else [0, 0, 0]
                    }
                    for ext in ext_np
                ]
            else:  # Single [3, 4]
                data["extrinsics"] = [{
                    "matrix": ext_np.tolist(),
                    "format": "opencv_w2c",
                    "rotation": ext_np[:3, :3].tolist(),
                    "translation": ext_np[:3, 3].tolist() if ext_np.shape[1] > 3 else [0, 0, 0]
                }]

        # Serialize intrinsics
        if intrinsics is not None:
            int_np = intrinsics.cpu().numpy()

            if int_np.ndim == 3:  # Batch [N, 3, 3]
                data["intrinsics"] = [
                    {
                        "matrix": intr.tolist(),
                        "fx": float(intr[0, 0]),
                        "fy": float(intr[1, 1]),
                        "cx": float(intr[0, 2]),
                        "cy": float(intr[1, 2])
                    }
                    for intr in int_np
                ]
            else:  # Single [3, 3]
                data["intrinsics"] = [{
                    "matrix": int_np.tolist(),
                    "fx": float(int_np[0, 0]),
                    "fy": float(int_np[1, 1]),
                    "cx": float(int_np[0, 2]),
                    "cy": float(int_np[1, 2])
                }]

        # Add depth statistics
        if raw_depth is not None:
            depth_np = raw_depth.cpu().numpy()
            data["depth_stats"] = {
                "min": float(depth_np.min()),
                "max": float(depth_np.max()),
                "mean": float(depth_np.mean()),
                "std": float(depth_np.std())
            }

        # Add confidence statistics if available
        if confidence is not None:
            conf_np = confidence.cpu().numpy()
            data["confidence_stats"] = {
                "mean": float(conf_np.mean()),
                "std": float(conf_np.std()),
                "min": float(conf_np.min()),
                "max": float(conf_np.max())
            }

        return json.dumps(data, indent=2)

    def estimate_depth(self, 
                     image: torch.Tensor, 
                     model_name: str, 
                     input_size: int = 1024,
                     blur_radius: int = 0, 
                     median_size: int = 0, 
                     apply_auto_contrast: bool = False, 
                     apply_gamma: bool = False,
                     force_reload: bool = False,
                     force_cpu: bool = False,
                     enable_camera_estimation: bool = True,
                     output_raw_depth: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Estimates depth from input image with error handling and cleanup.
        
        Args:
            image: Input image tensor
            model_name: Name of the depth model to use
            input_size: Target size for the longest dimension of the image (between 384 and 8192)
            blur_radius: Gaussian blur radius for smoothing
            median_size: Size of median filter for noise reduction
            apply_auto_contrast: Whether to enhance contrast automatically
            apply_gamma: Whether to apply gamma correction
            force_reload: Whether to force reload the model
            force_cpu: Whether to force using CPU for inference
            enable_camera_estimation: Whether to output camera intrinsics/extrinsics (for DA3 models)
            output_raw_depth: Whether to output the raw, unnormalized depth map (for DA3 models)
            
        Returns:
            Tuple containing:
            - depth: Processed depth map tensor (1, H, W, 3)
            - confidence: Confidence map tensor (1, H, W, 3) or None
            - extrinsics: Camera extrinsics tensor (1, 3, 4) or None
            - intrinsics: Camera intrinsics tensor (1, 3, 3) or None
            - camera_json: JSON string of camera parameters or empty string
        """
        error_image = None
        camera_data = {}
        start_time = time.time()
        
        # Initialize all return values to None or empty
        depth_output = None
        # Get input dimensions for dynamic placeholder sizing
        if image is not None and hasattr(image, 'shape') and len(image.shape) >= 3:
            img_h, img_w = image.shape[1], image.shape[2]
        else:
            img_h, img_w = 512, 512  # Fallback only if input is invalid
        confidence_output = torch.zeros((1, img_h, img_w, 3), dtype=torch.float32, device=image.device if hasattr(image, 'device') else 'cpu')
        extrinsics_output = None
        intrinsics_output = None
        camera_json_output = ""

        try:
            # Sanity check inputs and log initial info
            logger.info(f"Starting depth estimation with model: {model_name}, input_size: {input_size}, force_cpu: {force_cpu}, enable_camera_estimation: {enable_camera_estimation}, output_raw_depth: {output_raw_depth}")
            
            # Enhanced input validation with better error handling
            if image is None:
                logger.error("Input image is None")
                error_image = self._create_basic_error_image()
                self._add_error_text_to_image(error_image, "Input image is None")
                return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
            
            if image.numel() == 0:
                logger.error("Input image is empty (zero elements)")
                error_image = self._create_basic_error_image()
                self._add_error_text_to_image(error_image, "Input image is empty (zero elements)")
                return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
            
            # Log tensor information before processing
            logger.info(f"Input tensor shape: {image.shape}, dtype: {image.dtype}, device: {image.device}")
            
            # Verify tensor dimensions - support different input formats
            if image.ndim != 4:
                logger.warning(f"Expected 4D tensor for image, got {image.ndim}D. Attempting to reshape.")
                try:
                    # Try to reshape based on common dimension patterns
                    if image.ndim == 3:
                        # Could be [C, H, W] or [H, W, C] format
                        if image.shape[0] <= 3 and image.shape[0] > 0:  # Likely [C, H, W]
                            image = image.unsqueeze(0)  # Add batch dim -> [1, C, H, W]
                            logger.info(f"Reshaped 3D tensor to 4D with shape: {image.shape}")
                        elif image.shape[-1] <= 3 and image.shape[-1] > 0:  # Likely [H, W, C]
                            # Permute to [C, H, W] then add batch dim
                            image = image.permute(2, 0, 1).unsqueeze(0)
                            logger.info(f"Reshaped HWC tensor to BCHW with shape: {image.shape}")
                        else:
                            # Assume single channel image and add missing dimensions
                            image = image.unsqueeze(0).unsqueeze(0)
                            logger.info(f"Added batch and channel dimensions to 3D tensor: {image.shape}")
                    elif image.ndim == 2:
                        # Assume [H, W] format - add batch and channel dims
                        image = image.unsqueeze(0).unsqueeze(0)
                        logger.info(f"Reshaped 2D tensor to 4D with shape: {image.shape}")
                    elif image.ndim > 4:
                        # Too many dimensions, collapse extras
                        orig_shape = image.shape
                        image = image.reshape(1, orig_shape[1], orig_shape[-2], orig_shape[-1])
                        logger.info(f"Collapsed >4D tensor to 4D with shape: {image.shape}")
                    else:
                        # Fallback for other unusual dimensions
                        logger.error(f"Cannot automatically reshape tensor with {image.ndim} dimensions")
                        error_image = self._create_basic_error_image()
                        self._add_error_text_to_image(error_image, f"Unsupported tensor dimensions: {image.ndim}D")
                        return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
                except Exception as reshape_error:
                    logger.error(f"Error reshaping tensor: {str(reshape_error)}")
                    error_image = self._create_basic_error_image()
                    self._add_error_text_to_image(error_image, f"Error reshaping tensor: {str(reshape_error)[:100]}")
                    return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
            
            # Comprehensive type checking and conversion - verify at multiple points
            # 1. Initial type check and convert if needed
            if image.dtype != torch.float32:
                logger.info(f"Converting input tensor from {image.dtype} to torch.float32")
                
                # Safe conversion that handles different input types
                try:
                    if image.dtype == torch.uint8:
                        # Normalize [0-255] -> [0-1] for uint8 inputs
                        image = image.float() / 255.0
                    else:
                        # Standard conversion for other types
                        image = image.float()
                except Exception as type_error:
                    logger.error(f"Error converting tensor type: {str(type_error)}")
                    error_image = self._create_basic_error_image()
                    self._add_error_text_to_image(error_image, f"Type conversion error: {str(type_error)[:100]}")
                    return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
            
            # Create error image placeholder based on input dimensions
            error_image = self._create_error_image(image)
            
            # 2. Check for NaN/Inf values and fix them
            nan_count = torch.isnan(image).sum().item()
            inf_count = torch.isinf(image).sum().item()
            
            if nan_count > 0 or inf_count > 0:
                logger.warning(f"Input contains {nan_count} NaN and {inf_count} Inf values. Fixing problematic values.")
                image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 3. Check value range and normalize if needed
            min_val, max_val = image.min().item(), image.max().item()
            if max_val > 1.0 + 1e-5:  # Allow small floating point error
                logger.info(f"Input values outside [0,1] range: min={min_val}, max={max_val}. Normalizing.")
                
                # Ensure values are in [0,1] range - handle different input formats
                if min_val >= 0 and max_val <= 255:
                    # Likely [0-255] range
                    image = image / 255.0
                else:
                    # General min-max normalization
                    image = (image - min_val) / (max_val - min_val)
            
            # Parameter validation with safer defaults
            # Validate and normalize median_size parameter
            median_size_int = int(median_size) # Already int from INPUT_TYPES
            if median_size_int < 0:
                logger.warning(f"Invalid median_size: '{median_size}'. Using default '0'.")
                median_size_int = 0
            
            # Validate input_size with stricter bounds
            if not isinstance(input_size, (int, float)):
                logger.warning(f"Invalid input_size type: {type(input_size)}. Using default 1024.")
                input_size = 1024
            else:
                # Convert to int and constrain to valid range
                try:
                    input_size = int(input_size)
                    input_size = max(384, min(input_size, 8192))  # Clamp between 384 and 8192
                except:
                    logger.warning(f"Error converting input_size to int. Using default 1024.")
                    input_size = 1024
            
            # Try loading the model with graceful fallback
            try:
                self.ensure_model_loaded(model_name, force_reload, force_cpu)
                logger.info(f"Model '{model_name}' loaded successfully")
            except Exception as model_error:
                error_msg = f"Failed to load model '{model_name}': {str(model_error)}"
                logger.error(error_msg)
                
                # Try a more reliable fallback model before giving up
                fallback_models = ["MiDaS-Small", "MiDaS-Base"]
                for fallback_model in fallback_models:
                    if fallback_model != model_name:
                        logger.info(f"Attempting to load fallback model: {fallback_model}")
                        try:
                            self.ensure_model_loaded(fallback_model, True, True)  # Force reload and CPU for reliability
                            logger.info(f"Fallback model '{fallback_model}' loaded successfully")
                            # Update model_name to reflect the fallback
                            model_name = fallback_model
                            # Break the loop since we successfully loaded a fallback
                            break
                        except Exception as fallback_error:
                            logger.warning(f"Fallback model '{fallback_model}' also failed: {str(fallback_error)}")
                            continue
                
                # If we still don't have a model loaded, return error image
                if self.depth_estimator is None:
                    self._add_error_text_to_image(error_image, f"Model Error: {str(model_error)[:100]}...")
                    return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
            
            # Process input image with enhanced error recovery
            try:
                # Convert to PIL with robust error handling
                pil_image = self.process_image(image, input_size)
                # Store original dimensions for later resizing
                original_width, original_height = pil_image.size
                logger.info(f"Image processed to size: {pil_image.size} (will preserve these dimensions in output)")
            except Exception as img_error:
                logger.error(f"Image processing error: {str(img_error)}")
                logger.error(traceback.format_exc())
                
                # Try a more basic approach if the standard processing fails
                try:
                    logger.info("Attempting basic image conversion as fallback")
                    # Simple conversion fallback
                    if image.shape[1] > 3:  # BCHW format with unusual channel count
                        # Select first 3 channels or average if > 3
                        logger.warning(f"Unusual channel count: {image.shape[1]}. Using first 3 channels.")
                        if image.shape[1] > 3:
                            image = image[:, :3, :, :]
                    
                    # Convert to CPU numpy array
                    img_np = image.squeeze(0).cpu().numpy()
                    
                    # Handle different layouts
                    if img_np.shape[0] <= 3:  # [C, H, W]
                        img_np = np.transpose(img_np, (1, 2, 0))  # -> [H, W, C]
                    
                    # Ensure 3 channels
                    if len(img_np.shape) == 2:  # Grayscale
                        img_np = np.stack([img_np] * 3, axis=-1)
                    elif img_np.shape[-1] == 1:  # Single channel
                        img_np = np.concatenate([img_np] * 3, axis=-1)
                    
                    # Normalize values
                    if img_np.max() > 1.0:
                        img_np = img_np / 255.0
                    
                    # Convert to PIL
                    img_np = (img_np * 255).astype(np.uint8)
                    pil_image = Image.fromarray(img_np)
                    
                    # Store original dimensions
                    original_width, original_height = pil_image.size
                    
                    # Resize to appropriate dimensions
                    if input_size > 0:
                        w, h = pil_image.size
                        # Determine which dimension to scale to input_size
                        if w > h:
                            new_w = input_size
                            new_h = int(h * (new_w / w))
                        else:
                            new_h = input_size
                            new_w = int(w * (new_h / h))
                        
                        # Ensure dimensions are multiples of 32
                        new_h = ((new_h + 31) // 32) * 32
                        new_w = ((new_w + 31) // 32) * 32
                        
                        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
                    
                    logger.info(f"Fallback image processing succeeded with size: {pil_image.size}")
                except Exception as fallback_error:
                    logger.error(f"Fallback image processing also failed: {str(fallback_error)}")
                    self._add_error_text_to_image(error_image, f"Image Error: {str(img_error)[:100]}...")
                    return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
            
            # Depth estimation with comprehensive error handling
            try:
                # Use inference_mode for better memory usage
                with torch.inference_mode():
                    logger.info(f"Running inference on image of size {pil_image.size}")
                    
                    # Ensure the depth estimator is in eval mode
                    if hasattr(self.depth_estimator, 'eval'):
                        self.depth_estimator.eval()
                    
                    # Add timing for performance analysis
                    inference_start = time.time()
                    
                    # Perform inference with better error handling
                    try:
                        depth_result = self.depth_estimator(pil_image)
                        inference_time = time.time() - inference_start
                        logger.info(f"Depth inference completed in {inference_time:.2f} seconds")
                    except Exception as inference_error:
                        logger.error(f"Depth estimator inference error: {str(inference_error)}")
                        
                        # Detailed error analysis for better debugging
                        error_str = str(inference_error).lower()
                        
                        # Try fallback for common error types
                        if "cuda" in error_str and "out of memory" in error_str:
                            logger.warning("CUDA out of memory detected. Attempting CPU fallback.")
                            # Already on CPU? Check and handle
                            if force_cpu:
                                logger.error("Already using CPU but still encountered memory error")
                                self._add_error_text_to_image(error_image, "Memory error even on CPU. Try smaller input size.")
                                return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
                            else:
                                # Try CPU fallback
                                logger.info("Switching to CPU processing")
                                return self.estimate_depth(
                                    image.cpu(), model_name, input_size, blur_radius, median_size,
                                    apply_auto_contrast, apply_gamma, True, True, enable_camera_estimation, output_raw_depth  # Force CPU
                                )
                        
                        # Type mismatch errors
                        elif "input type" in error_str and "weight type" in error_str:
                            logger.warning("Tensor type mismatch detected. Attempting explicit type conversion.")
                            # Try with explicit CPU conversion
                            return self.estimate_depth(
                                image.float().cpu(), model_name, input_size, blur_radius, median_size,
                                apply_auto_contrast, apply_gamma, True, True, enable_camera_estimation, output_raw_depth  # Force CPU and reload
                            )
                        
                        # Dimension mismatch errors
                        elif "dimensions" in error_str or "dimension" in error_str or "shape" in error_str:
                            logger.warning("Tensor dimension mismatch detected. Trying alternate approach.")
                            # Fall back to CPU MiDaS model which has more robust dimension handling
                            logger.info("Falling back to MiDaS model on CPU")
                            try:
                                # Clean up current model
                                self.cleanup()
                                # Try to load MiDaS model
                                self.ensure_model_loaded("MiDaS-Small", True, True)
                                # Retry with the new model
                                return self.estimate_depth(
                                    image.cpu(), "MiDaS-Small", input_size, blur_radius, median_size,
                                    apply_auto_contrast, apply_gamma, False, True, enable_camera_estimation, output_raw_depth  # Already reloaded, force CPU
                                )
                            except Exception as midas_error:
                                logger.error(f"MiDaS fallback also failed: {str(midas_error)}")
                                self._add_error_text_to_image(error_image, f"Inference Error: {str(inference_error)[:100]}...")
                                return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
                        
                        # Other errors - just return error image
                        self._add_error_text_to_image(error_image, f"Inference Error: {str(inference_error)[:100]}...")
                        return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
                    
                    # Handle DA3 specific output
                    if isinstance(depth_result, DA3Prediction):
                        predicted_depth_tensor = depth_result.depth
                        if output_raw_depth and depth_result.raw_depth is not None:
                            # If raw depth is requested and available, use it for the main depth output
                            # Ensure it's normalized to 0-1 for display purposes, but keep its original range for camera data
                            raw_depth_for_output = depth_result.raw_depth.squeeze(0).cpu().numpy()
                            raw_depth_min, raw_depth_max = raw_depth_for_output.min(), raw_depth_for_output.max()
                            if raw_depth_max - raw_depth_min > 1e-6:
                                depth_map = (raw_depth_for_output - raw_depth_min) / (raw_depth_max - raw_depth_min)
                            else:
                                depth_map = np.zeros_like(raw_depth_for_output)
                        else:
                            depth_map = predicted_depth_tensor.squeeze(0).cpu().numpy()

                        if depth_result.confidence is not None:
                            conf = depth_result.confidence
                            # Ensure batch dimension [N, H, W]
                            if conf.ndim == 2:
                                conf = conf.unsqueeze(0)  # [H, W] -> [1, H, W]
                            # Convert to RGB format [N, H, W, 3] for ComfyUI IMAGE type
                            confidence_output = conf.unsqueeze(-1).repeat(1, 1, 1, 3)  # [N, H, W, 3]

                        if enable_camera_estimation and depth_result.supports_pose:
                            if depth_result.extrinsics is not None:
                                ext = depth_result.extrinsics
                                # Ensure batch dimension for single image case
                                if ext.ndim == 2:  # [3, 4] -> [1, 3, 4]
                                    extrinsics_output = ext.unsqueeze(0)
                                else:  # Already [N, 3, 4]
                                    extrinsics_output = ext
                            if depth_result.intrinsics is not None:
                                intr = depth_result.intrinsics
                                # Ensure batch dimension for single image case
                                if intr.ndim == 2:  # [3, 3] -> [1, 3, 3]
                                    intrinsics_output = intr.unsqueeze(0)
                                else:  # Already [N, 3, 3]
                                    intrinsics_output = intr
                            
                            # Prepare camera_json_output
                            # Prepare camera_json_output using the helper method
                            try:
                                camera_json_output = self._serialize_camera_data(
                                    extrinsics=depth_result.extrinsics,
                                    intrinsics=depth_result.intrinsics,
                                    confidence=depth_result.confidence,
                                    raw_depth=depth_result.raw_depth,
                                    model_name=model_name,
                                    image_width=original_width,
                                    image_height=original_height
                                )
                                logger.info(f"Camera JSON generated successfully. Length: {len(camera_json_output)}")
                                # Explicit print for user debugging as requested
                                print(f"DA3 Camera JSON Output (first 200 chars): {camera_json_output[:200]}...")
                            except Exception as json_error:
                                logger.error(f"Error serializing camera data: {str(json_error)}")
                                logger.error(traceback.format_exc())
                                camera_json_output = json.dumps({"error": str(json_error)})
                        else:
                            logger.info("Camera estimation disabled or not supported by model.")
                            # Still output informative JSON even without camera data
                            camera_json_output = json.dumps({
                                "status": "camera_estimation_unavailable",
                                "reason": "disabled" if not enable_camera_estimation else "model_does_not_support_pose",
                                "model": model_name,
                                "supports_pose": depth_result.supports_pose,
                                "enable_camera_estimation": enable_camera_estimation,
                                "extrinsics_available": depth_result.extrinsics is not None,
                                "intrinsics_available": depth_result.intrinsics is not None,
                                "image_width": original_width,
                                "image_height": original_height
                            }, indent=2)

                    else: # Handle V1/V2/MiDaS pipeline output
                        # V1/V2/MiDaS don't support camera estimation - output informative JSON
                        camera_json_output = json.dumps({
                            "status": "not_supported",
                            "reason": "model_type_does_not_support_camera_estimation",
                            "model": model_name,
                            "message": "Camera pose estimation requires Depth-Anything-V3 models",
                            "image_width": original_width,
                            "image_height": original_height
                        }, indent=2)
                        # Verify depth result and convert to float32
                        if not isinstance(depth_result, dict) or "predicted_depth" not in depth_result:
                            logger.error(f"Invalid depth result format: {type(depth_result)}")
                            self._add_error_text_to_image(error_image, "Invalid depth result format")
                            return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
                        
                        # Extract and validate predicted depth
                        predicted_depth_tensor = depth_result["predicted_depth"]
                        
                        # Ensure correct tensor type
                        if not torch.is_tensor(predicted_depth_tensor):
                            logger.error(f"Predicted depth is not a tensor: {type(predicted_depth_tensor)}")
                            self._add_error_text_to_image(error_image, "Predicted depth is not a tensor")
                            return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
                        
                        # Convert to float32 if needed
                        if predicted_depth_tensor.dtype != torch.float32:
                            logger.info(f"Converting predicted depth from {predicted_depth_tensor.dtype} to float32")
                            predicted_depth_tensor = predicted_depth_tensor.float()
                        
                        # Convert to CPU for post-processing
                        depth_map = predicted_depth_tensor.squeeze().cpu().numpy()

            except RuntimeError as rt_error:
                # Handle runtime errors separately for clearer error messages
                error_msg = str(rt_error)
                logger.error(f"Runtime error during depth estimation: {error_msg}")
                
                # Check for specific error types
                if "CUDA out of memory" in error_msg:
                    logger.warning("CUDA out of memory. Trying CPU fallback.")
                    
                    # Only try CPU fallback if not already using CPU
                    if not force_cpu:
                        try:
                            logger.info("Switching to CPU processing")
                            return self.estimate_depth(
                                image.cpu(), model_name, input_size, blur_radius, median_size,
                                apply_auto_contrast, apply_gamma, True, True, enable_camera_estimation, output_raw_depth  # Force reload and CPU
                            )
                        except Exception as cpu_error:
                            logger.error(f"CPU fallback failed: {str(cpu_error)}")
                    
                    self._add_error_text_to_image(error_image, "CUDA Out of Memory. Try a smaller model or image size.")
                else:
                    # Generic runtime error
                    self._add_error_text_to_image(error_image, f"Runtime Error: {error_msg[:100]}...")
                
                return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
            except Exception as e:
                # Handle other exceptions
                error_msg = f"Depth estimation failed: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                self._add_error_text_to_image(error_image, f"Error: {str(e)[:100]}...")
                return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
            
            # Validate depth map
            # Check for NaN/Inf values
            if np.isnan(depth_map).any() or np.isinf(depth_map).any():
                logger.warning("Depth map contains NaN or Inf values. Replacing with zeros.")
                depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Check for empty or invalid depth map
            if depth_map.size == 0:
                logger.error("Depth map is empty")
                self._add_error_text_to_image(error_image, "Empty depth map returned")
                return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
            
            # Post-processing with enhanced error handling
            try:
                # Ensure depth values have reasonable range for normalization
                depth_min, depth_max = depth_map.min(), depth_map.max()
                
                # Handle constant depth maps (avoid division by zero)
                if np.isclose(depth_max, depth_min):
                    logger.warning("Constant depth map detected (min = max). Using values directly.")
                    # Just use a normalized constant value
                    depth_map = np.ones_like(depth_map) * 0.5
                else:
                    # Normalize to [0, 1] range first - safer for later operations
                    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
                
                # Scale to [0, 255] for PIL operations
                depth_map_uint8 = (depth_map * 255.0).astype(np.uint8)
                
                # Log the depth map shape coming from the model
                logger.info(f"Depth map shape from model: {depth_map_uint8.shape}")
                
                # Create PIL image explicitly with L mode (grayscale)
                try:
                    depth_pil = Image.fromarray(depth_map_uint8, mode='L')
                    logger.info(f"Depth PIL image size before resize: {depth_pil.size}")
                except Exception as pil_error:
                    logger.error(f"Error creating PIL image: {str(pil_error)}")
                    # Try to reshape the array if dimensions are wrong
                    if len(depth_map_uint8.shape) != 2:
                        logger.warning(f"Unexpected depth map shape: {depth_map_uint8.shape}. Attempting to fix.")
                        # Try to extract first channel if multi-channel
                        if len(depth_map_uint8.shape) > 2:
                            depth_map_uint8 = depth_map_uint8[..., 0]
                        elif len(depth_map_uint8.shape) == 1:
                            # 1D array - try to reshape to 2D
                            h = int(np.sqrt(depth_map_uint8.size))
                            w = depth_map_uint8.size // h
                            depth_map_uint8 = depth_map_uint8.reshape(h, w)
                        
                        depth_pil = Image.fromarray(depth_map_uint8, mode='L')
                
                # Resize depth map to original dimensions from input image before post-processing
                # This is the key fix for the resolution issue
                try:
                    logger.info(f"Resizing depth map to original dimensions: {original_width}x{original_height}")
                    depth_pil = depth_pil.resize((original_width, original_height), Image.BICUBIC)
                    logger.info(f"Depth PIL image size after resize: {depth_pil.size}")
                except Exception as resize_error:
                    logger.error(f"Error resizing depth map: {str(resize_error)}")
                    logger.error(traceback.format_exc())
                
                # Apply post-processing with parameter validation
                # Apply blur if radius is positive
                if blur_radius > 0:
                    try:
                        depth_pil = depth_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                    except Exception as blur_error:
                        logger.warning(f"Error applying blur: {str(blur_error)}. Skipping.")
                
                # Apply median filter if size is valid (must be odd number > 1)
                try:
                    median_size_int = int(median_size) if median_size is not None else 0
                    # Ensure median size is odd (PIL requirement) - round up to nearest odd
                    if median_size_int > 1:
                        if median_size_int % 2 == 0:
                            median_size_int += 1  # Make odd
                        depth_pil = depth_pil.filter(ImageFilter.MedianFilter(size=median_size_int))
                except Exception as median_error:
                    logger.warning(f"Error applying median filter: {str(median_error)}. Skipping.")
                
                # Apply auto contrast if requested
                if apply_auto_contrast:
                    try:
                        depth_pil = ImageOps.autocontrast(depth_pil)
                    except Exception as contrast_error:
                        logger.warning(f"Error applying auto contrast: {str(contrast_error)}. Skipping.")
                
                # Apply gamma correction if requested
                if apply_gamma:
                    try:
                        depth_array = np.array(depth_pil).astype(np.float32) / 255.0
                        mean_luminance = np.mean(depth_array)
                        
                        # Avoid division by zero or negative values
                        if mean_luminance > 0.001:
                            # Calculate gamma based on mean luminance for adaptive correction
                            gamma = np.log(0.5) / np.log(mean_luminance)
                            
                            # Clamp gamma to reasonable range to avoid extreme corrections
                            gamma = max(0.1, min(gamma, 3.0))
                            logger.info(f"Applying gamma correction with value: {gamma:.2f}")
                            
                            # Apply gamma correction
                            corrected = np.power(depth_array, 1.0/gamma) * 255.0
                            depth_pil = Image.fromarray(corrected.astype(np.uint8), mode='L')
                        else:
                            logger.warning(f"Mean luminance too low: {mean_luminance}. Skipping gamma correction.")
                    except Exception as gamma_error:
                        logger.warning(f"Error applying gamma correction: {str(gamma_error)}. Skipping.")
                
                # Convert processed image back to tensor
                # Convert to numpy array for tensor conversion
                depth_array = np.array(depth_pil).astype(np.float32) / 255.0
                
                # Final validation checks
                # Check for invalid dimensions
                h, w = depth_array.shape
                if h <= 1 or w <= 1:
                    logger.error(f"Invalid depth map dimensions: {h}x{w}")
                    self._add_error_text_to_image(error_image, "Invalid depth map dimensions (too small)")
                    return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
                
                # Log final dimensions for debugging
                logger.info(f"Final depth map dimensions: {h}x{w}")
                
                # Create RGB depth map by stacking the same grayscale image three times
                # Stack to create a 3-channel image compatible with ComfyUI
                depth_rgb = np.stack([depth_array] * 3, axis=-1)  # Shape becomes (h, w, 3)
                
                # Convert to tensor and add batch dimension
                depth_tensor = torch.from_numpy(depth_rgb).unsqueeze(0).float()  # Shape becomes (1, h, w, 3)
                
                # Optional: move to device if not using CPU
                if self.device is not None and not force_cpu:
                    depth_tensor = depth_tensor.to(self.device)
                
                # Validate output tensor
                if torch.isnan(depth_tensor).any() or torch.isinf(depth_tensor).any():
                    logger.warning("Final tensor contains NaN or Inf values. Fixing.")
                    depth_tensor = torch.nan_to_num(depth_tensor, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Ensure values are in [0, 1] range
                min_val, max_val = depth_tensor.min().item(), depth_tensor.max().item()
                if min_val < 0.0 or max_val > 1.0:
                    logger.warning(f"Depth tensor values outside [0,1] range: min={min_val}, max={max_val}. Normalizing.")
                    depth_tensor = torch.clamp(depth_tensor, 0.0, 1.0)
                
                # Log completion info
                processing_time = time.time() - start_time
                logger.info(f"Depth processing completed in {processing_time:.2f} seconds")
                logger.info(f"Output tensor: shape={depth_tensor.shape}, dtype={depth_tensor.dtype}, device={depth_tensor.device}")

                return (depth_tensor, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
                
            except Exception as post_error:
                # Handle post-processing errors
                error_msg = f"Error during depth map post-processing: {str(post_error)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                self._add_error_text_to_image(error_image, f"Post-processing Error: {str(post_error)[:100]}...")
                return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
                
        except Exception as e:
            # Global catch-all error handler
            error_msg = f"Depth estimation failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Create error image if needed
            if error_image is None:
                error_image = self._create_basic_error_image()
                
            self._add_error_text_to_image(error_image, f"Unexpected Error: {str(e)[:100]}...")
            return (error_image, confidence_output, extrinsics_output, intrinsics_output, camera_json_output)
        finally:
            # Always clean up resources regardless of success or failure
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