---
name: comfyui-node-development
description: "This skill should be used when developing ComfyUI custom nodes, creating new node types, implementing AI model integrations, or extending ComfyUI functionality with Python-based nodes."
category: ai-ml
risk: safe
source: community
tags: "[comfyui, ai, stable-diffusion, nodes, python, pytorch]"
date_added: "2026-03-09"
---

# ComfyUI Node Development

## Purpose

Build production-ready ComfyUI custom nodes following official best practices. This skill covers node architecture, input/output handling, execution patterns, model loading, and UI integration for ComfyUI's node-based AI workflow system.

## When to Use This Skill

This skill should be used when:
- Creating new custom nodes for ComfyUI
- Implementing AI model inference pipelines
- Adding custom preprocessing or postprocessing nodes
- Developing nodes for Stable Diffusion workflows
- Extending ComfyUI with third-party integrations
- Debugging node execution issues
- Optimizing node performance for GPU/CPU execution

## Core Capabilities

1. **Node Architecture** - ComfyUI's node class structure and inheritance patterns
2. **Input/Output Types** - ComfyUI type system (IMAGE, LATENT, CONDITIONING, MODEL, etc.)
3. **Execution Model** - Understanding the graph execution and caching mechanism
4. **Model Management** - Loading, caching, and managing AI models in nodes
5. **UI Integration** - Custom widgets, dropdowns, and node visual customization
6. **Error Handling** - Graceful error reporting and recovery in nodes
7. **Performance Optimization** - Memory management, batching, and GPU optimization

## Node Development Fundamentals

### Basic Node Structure

Every ComfyUI node follows this structure:

```python
class MyCustomNode:
    """Node description that appears in the UI."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": ("INPUT_TYPE", {"default": value, "min": 0, "max": 100}),
            },
            "optional": {
                "optional_input": ("INPUT_TYPE", {"default": value}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    RETURN_TYPES = ("OUTPUT_TYPE",)
    RETURN_NAMES = ("output_name",)
    FUNCTION = "execute"
    CATEGORY = "custom/category"
    
    def execute(self, input_name, optional_input=None, prompt=None, extra_pnginfo=None):
        # Node logic here
        return (output,)
```

### Input Types Reference

| Type | Description | Example Widget |
|------|-------------|----------------|
| `IMAGE` | Torch tensor (B, H, W, C) | Image input/output |
| `LATENT` | Latent representation | Latent input/output |
| `MODEL` | Diffusion model object | Model loader output |
| `CLIP` | CLIP model for conditioning | CLIP loader output |
| `CONDITIONING` | Text conditioning data | CLIP text encode output |
| `VAE` | VAE model for encode/decode | VAE loader output |
| `MASK` | Single channel image | Mask input/output |
| `INT` | Integer value | Number widget |
| `FLOAT` | Float value | Float widget |
| `STRING` | Text string | Text input |
| `BOOLEAN` | True/False | Toggle widget |

### Widget Configurations

```python
# Integer with constraints
"steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1})

# Float with slider
"denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

# String with multiline
"prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

# Dropdown/Combo
"mode": (["option1", "option2", "option3"], {"default": "option1"})

# Boolean toggle
"enabled": ("BOOLEAN", {"default": True})
```

## Implementation Patterns

### Image Processing Node

```python
import torch
import numpy as np
from PIL import Image

class ImagePreprocessor:
    """Preprocess images for model input."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "interpolation": (["nearest", "bilinear", "bicubic"], {"default": "bilinear"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    FUNCTION = "preprocess"
    CATEGORY = "image/preprocessing"
    
    def preprocess(self, image, target_width, target_height, interpolation):
        # image is (B, H, W, C) tensor in range [0, 1]
        batch_size, height, width, channels = image.shape
        
        # Convert to PIL for resizing
        images = []
        for i in range(batch_size):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Resize
            pil_img = pil_img.resize((target_width, target_height), 
                                     getattr(Image, interpolation.upper()))
            
            # Convert back to tensor
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            images.append(torch.from_numpy(img_np))
        
        result = torch.stack(images)
        return (result,)
```

### Model Wrapper Node

```python
import comfy.model_management as model_management
import folder_paths

class CustomModelLoader:
    """Load custom models with proper memory management."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_model"
    CATEGORY = "loaders"
    
    def load_model(self, model_name, device):
        # Get full path
        model_path = folder_paths.get_full_path("checkpoints", model_name)
        
        # Load with ComfyUI's model management
        if device == "auto":
            device = model_management.get_torch_device()
        
        # Load checkpoint (implementation depends on model type)
        # Use comfy.utils.load_checkpoint_guess_config or similar
        
        return (model, clip, vae)
```

### Conditional Execution Node

```python
class ConditionalImageProcessor:
    """Process images conditionally based on inputs."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/processing"
    
    def process(self, image, enabled, strength=1.0):
        if not enabled:
            return (image,)  # Pass through
        
        # Apply processing
        processed = image * strength
        processed = torch.clamp(processed, 0, 1)
        
        return (processed,)
```

## Advanced Patterns

### Node with Side Effects

```python
import json
import os

class SaveMetadata:
    """Save workflow metadata alongside images."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True  # Mark as output node
    FUNCTION = "save"
    CATEGORY = "output"
    
    def save(self, images, filename_prefix, prompt=None, extra_pnginfo=None):
        # Save logic here
        full_output_folder = folder_paths.get_output_directory()
        
        for batch_number, image in enumerate(images):
            # Save image
            # Save metadata JSON
            if extra_pnginfo is not None:
                metadata = {
                    "prompt": prompt,
                    "workflow": extra_pnginfo.get("workflow", {})
                }
                # Write metadata file
        
        return {}  # Empty return for output nodes
```

### Batch Processing Node

```python
class BatchImageProcessor:
    """Process multiple images efficiently."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "operation": (["normalize", "invert", "grayscale"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_batch"
    CATEGORY = "image/batch"
    
    def process_batch(self, images, operation):
        # Process entire batch at once (GPU accelerated)
        if operation == "normalize":
            mean = images.mean(dim=(1, 2, 3), keepdim=True)
            std = images.std(dim=(1, 2, 3), keepdim=True)
            result = (images - mean) / (std + 1e-8)
        
        elif operation == "invert":
            result = 1.0 - images
        
        elif operation == "grayscale":
            # RGB to grayscale
            weights = torch.tensor([0.299, 0.587, 0.114], device=images.device)
            gray = (images * weights.view(1, 1, 1, 3)).sum(dim=-1, keepdim=True)
            result = gray.expand(-1, -1, -1, 3)
        
        return (torch.clamp(result, 0, 1),)
```

## Best Practices

### Memory Management

```python
import comfy.model_management as mm

class MemoryEfficientNode:
    def process(self, model, latent):
        # Get the appropriate device
        device = mm.get_torch_device()
        
        # Load to device only when needed
        model = model.to(device)
        latent = latent.to(device)
        
        # Process
        with torch.no_grad():
            result = model(latent)
        
        # Clean up if needed
        mm.soft_empty_cache()
        
        return (result,)
```

### Error Handling

```python
class RobustNode:
    def execute(self, image, factor):
        try:
            if image is None:
                raise ValueError("Input image is None")
            
            if factor <= 0:
                raise ValueError(f"Factor must be positive, got {factor}")
            
            result = self._process(image, factor)
            return (result,)
            
        except Exception as e:
            # Log error for debugging
            print(f"[RobustNode] Error: {e}")
            # Return input unchanged or raise
            raise
```

### Caching Considerations

ComfyUI caches node outputs based on inputs. For nodes with non-deterministic behavior:

```python
class RandomGenerator:
    """Generate random values with proper seed handling."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "generators"
    
    def generate(self, seed, width, height):
        # Use seed for reproducibility
        torch.manual_seed(seed)
        
        # Generate random noise
        noise = torch.randn(1, height, width, 3)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        return (noise,)
```

## Node Registration

Nodes must be registered in `__init__.py`:

```python
from .my_node_module import MyCustomNode, AnotherNode

NODE_CLASS_MAPPINGS = {
    "MyCustomNode": MyCustomNode,
    "AnotherNode": AnotherNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyCustomNode": "My Custom Node",
    "AnotherNode": "Another Node",
}
```

## Testing and Debugging

### Unit Testing Pattern

```python
def test_node():
    node = MyCustomNode()
    
    # Create test input
    test_image = torch.rand(1, 64, 64, 3)
    
    # Execute
    result = node.execute(test_image, factor=1.5)
    
    # Verify
    assert result[0].shape == test_image.shape
    assert result[0].min() >= 0 and result[0].max() <= 1
```

### Debug Logging

```python
import logging

logger = logging.getLogger(__name__)

class DebuggableNode:
    def execute(self, input_tensor):
        logger.debug(f"Input shape: {input_tensor.shape}")
        logger.debug(f"Input device: {input_tensor.device}")
        logger.debug(f"Input range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
        
        result = self.process(input_tensor)
        
        logger.debug(f"Output shape: {result.shape}")
        return (result,)
```

## Common Pitfalls

1. **Shape Mismatches**: Always verify tensor shapes match expected inputs/outputs
2. **Device Placement**: Ensure tensors are on the correct device (CPU/CUDA)
3. **Value Ranges**: Images should be [0, 1] range, not [0, 255]
4. **Batch Dimension**: Handle batch dimension (B) properly - nodes may receive batched inputs
5. **Memory Leaks**: Use `torch.no_grad()` for inference; clear caches when done
6. **Type Consistency**: RETURN_TYPES must match actual return values exactly

## References

- **ComfyUI Repository**: https://github.com/comfyanonymous/ComfyUI
- **ComfyUI Custom Nodes Guide**: https://docs.comfy.org/essentials/custom_nodes/
- **ComfyUI Node Examples**: https://github.com/comfyanonymous/ComfyUI/tree/master/custom_nodes
