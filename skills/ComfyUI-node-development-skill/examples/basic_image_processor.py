"""
Example: Basic Image Processor Node

A complete working example of a ComfyUI node that resizes images
with various interpolation methods.
"""

import torch
import numpy as np
from PIL import Image


class ExampleImageResizer:
    """
    Resize images to target dimensions with multiple interpolation options.
    
    This demonstrates:
    - Basic node structure
    - IMAGE input/output types
    - Dropdown widget configuration
    - PIL-based image processing
    - Batch processing support
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image tensor (B, H, W, C)"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Target width in pixels"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Target height in pixels"
                }),
                "interpolation": (["nearest", "bilinear", "bicubic", "lanczos"], {
                    "default": "bilinear",
                    "tooltip": "Resampling method"
                }),
            },
            "optional": {
                "maintain_aspect": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep original aspect ratio"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)
    FUNCTION = "resize"
    CATEGORY = "example/image"
    
    # Maps string names to PIL constants
    INTERPOLATION_MAP = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }
    
    def resize(self, image, width, height, interpolation, maintain_aspect=False):
        """
        Resize input image(s) to target dimensions.
        
        Args:
            image: Tensor of shape (B, H, W, C) with values in [0, 1]
            width: Target width
            height: Target height
            interpolation: Resampling method name
            maintain_aspect: Whether to preserve aspect ratio
            
        Returns:
            Tuple containing resized image tensor
        """
        # Get interpolation method
        interp_method = self.INTERPOLATION_MAP.get(interpolation, Image.BILINEAR)
        
        batch_size, orig_h, orig_w, channels = image.shape
        
        # Calculate dimensions if maintaining aspect ratio
        if maintain_aspect:
            aspect = orig_w / orig_h
            if width / height > aspect:
                width = int(height * aspect)
            else:
                height = int(width / aspect)
        
        # Process each image in batch
        resized_images = []
        for i in range(batch_size):
            # Convert tensor to PIL (0-255 range)
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Resize
            pil_img = pil_img.resize((width, height), interp_method)
            
            # Convert back to tensor (0-1 range)
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            resized_images.append(torch.from_numpy(img_np))
        
        # Stack back into batch
        result = torch.stack(resized_images)
        
        return (result,)


# Node registration (would go in __init__.py)
NODE_CLASS_MAPPINGS = {
    "ExampleImageResizer": ExampleImageResizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExampleImageResizer": "Example: Image Resizer",
}
