# ComfyUI Node Development - Comprehensive Guide

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Type System Deep Dive](#type-system-deep-dive)
3. [Execution Model](#execution-model)
4. [Model Management](#model-management)
5. [UI Customization](#ui-customization)
6. [Performance Optimization](#performance-optimization)
7. [Advanced Patterns](#advanced-patterns)

---

## Architecture Overview

### How ComfyUI Works

ComfyUI operates on a node-graph execution model:

1. **Graph Definition**: Users create workflows by connecting nodes
2. **Topological Sort**: Nodes are ordered based on dependencies
3. **Execution**: Each node executes when all inputs are ready
4. **Caching**: Outputs are cached based on input hashes for efficiency

### Node Lifecycle

```
Class Definition → Registration → Instantiation → Execution → Cleanup
```

### File Structure

```
my_custom_nodes/
├── __init__.py           # Node registration
├── nodes.py              # Node implementations
├── utils.py              # Helper functions
├── models/               # Model definitions
└── web/                  # JavaScript extensions (optional)
    └── my_extension.js
```

---

## Type System Deep Dive

### Core Types

#### IMAGE Type

```python
# IMAGE is a torch.Tensor with shape (B, H, W, C)
# Values are in range [0, 1] (float32)
# Channel order is RGB

# Converting to/from PIL
pil_image = Image.fromarray((tensor[0].numpy() * 255).astype(np.uint8))
tensor = torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0)
```

#### LATENT Type

```python
# LATENT is a dictionary containing:
# - "samples": torch.Tensor of shape (B, C, H, W)
# - "batch_index": Optional list for partial batch processing

latent = {
    "samples": torch.randn(1, 4, 64, 64),  # For SD 1.5
    "batch_index": [0, 1, 2]
}
```

#### MODEL Type

```python
# MODEL wraps the diffusion model
# Access the underlying model with model.model
# Use model.apply_model(x, t, c) for inference
```

#### CONDITIONING Type

```python
# CONDITIONING is a list of tuples:
# [(conditioning_vector, options_dict), ...]

cond = [
    (torch.randn(1, 77, 768), {"pooled_output": torch.randn(1, 768)})
]
```

### Custom Types

You can define custom types for type checking:

```python
# In your __init__.py or a types module
CUSTOM_TYPES = {
    "MY_CUSTOM_TYPE": "MY_CUSTOM_TYPE",
}

# Use in node
RETURN_TYPES = ("MY_CUSTOM_TYPE",)
```

---

## Execution Model

### Graph Execution Flow

```python
# Pseudo-code of ComfyUI execution
def execute_graph(graph, inputs):
    cache = {}
    executed = set()
    
    def execute_node(node_id):
        if node_id in executed:
            return cache[node_id]
        
        node = graph[node_id]
        
        # Execute dependencies first
        input_values = []
        for input_id in node.inputs:
            input_values.append(execute_node(input_id))
        
        # Execute this node
        result = node.function(*input_values)
        
        # Cache and return
        cache[node_id] = result
        executed.add(node_id)
        return result
    
    return execute_node(output_node_id)
```

### Forcing Re-execution

Nodes with random outputs should use unique inputs to bypass cache:

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "seed": ("INT", {"default": 0}),
        }
    }

# Different seed = different cache key = re-execution
```

---

## Model Management

### Model Loading

```python
import folder_paths
import comfy.utils

def load_checkpoint(self, ckpt_name):
    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
    
    out = comfy.utils.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    
    return out[:3]  # (model, clip, vae)
```

### Model Patching

```python
class ModelPatcher:
    """Apply patches to models (LoRA, etc.)"""
    
    def patch_model(self, model, patches):
        # Create patched copy
        model.patch_model(patches)
        return model
    
    def unpatch_model(self, model):
        # Remove patches
        model.unpatch_model()
```

### Memory Optimization

```python
import comfy.model_management as mm

class EfficientNode:
    def process(self, model, latent):
        device = mm.get_torch_device()
        
        # Move to device
        model = model.to(device)
        latent = latent.to(device)
        
        # Process with automatic memory management
        with mm.autocast():
            result = model(latent)
        
        # Optional: free memory
        mm.soft_empty_cache()
        
        return result
```

---

## UI Customization

### JavaScript Extensions

Create `web/my_extension.js`:

```javascript
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "MyCustomExtension",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MyCustomNode") {
            // Customize node appearance
            nodeType.prototype.onDrawForeground = function(ctx) {
                // Custom drawing code
            };
        }
    }
});
```

### Custom Widgets

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            # Color picker (requires JS extension)
            "color": ("COLOR", {"default": "#ff0000"}),
            
            # File upload
            "file": ("FILE", {"accept": ".png,.jpg"}),
            
            # Range slider
            "value": ("FLOAT", {
                "default": 0.5,
                "min": 0,
                "max": 1,
                "step": 0.01,
                "display": "slider"  # Requires JS extension
            }),
        }
    }
```

---

## Performance Optimization

### Batching

```python
def process_batch(self, images):
    # Process all at once instead of loop
    # Much faster on GPU
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    return images - mean
```

### In-Place Operations

```python
def modify_inplace(self, tensor):
    # Use in-place operations to save memory
    tensor.add_(0.1)  # In-place addition
    tensor.clamp_(0, 1)  # In-place clamp
    return tensor
```

### Mixed Precision

```python
import torch

def mixed_precision_process(self, model, x):
    with torch.cuda.amp.autocast():
        output = model(x)
    return output
```

---

## Advanced Patterns

### Dynamic Inputs

```python
class DynamicInputNode:
    """Node with variable number of inputs."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                f"input_{i}": ("IMAGE",) for i in range(10)
            }
        }
    
    def combine(self, **kwargs):
        # Collect all non-None inputs
        images = [v for k, v in kwargs.items() 
                  if k.startswith("input_") and v is not None]
        return (torch.cat(images, dim=0),)
```

### Node Composition

```python
class ComposedNode:
    """Combine multiple operations in one node."""
    
    def __init__(self):
        self.sub_node_1 = SubNode1()
        self.sub_node_2 = SubNode2()
    
    def execute(self, input_data):
        temp = self.sub_node_1.process(input_data)
        result = self.sub_node_2.process(temp)
        return (result,)
```

### Async Operations

```python
import asyncio

class AsyncNode:
    async def fetch_data(self, url):
        # Async HTTP request
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.read()
    
    def execute(self, url):
        # Run async in sync context
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(self.fetch_data(url))
        return (data,)
```

---

## Debugging Techniques

### Visual Debugging

```python
class DebugNode:
    """Print debug information about tensors."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("*",),  # Accept any type
                "print_stats": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("*",)
    FUNCTION = "debug"
    
    def debug(self, tensor, print_stats):
        print(f"=== DEBUG NODE ===")
        print(f"Type: {type(tensor)}")
        
        if isinstance(tensor, torch.Tensor):
            print(f"Shape: {tensor.shape}")
            print(f"Dtype: {tensor.dtype}")
            print(f"Device: {tensor.device}")
            print(f"Min/Max: {tensor.min():.4f} / {tensor.max():.4f}")
            print(f"Mean/Std: {tensor.mean():.4f} / {tensor.std():.4f}")
        elif isinstance(tensor, dict):
            print(f"Keys: {tensor.keys()}")
        
        print(f"==================")
        
        return (tensor,)
```

### Progress Reporting

```python
class ProgressNode:
    """Show progress for long operations."""
    
    def long_operation(self, items):
        from comfy.utils import ProgressBar
        
        pbar = ProgressBar(len(items))
        results = []
        
        for i, item in enumerate(items):
            result = self.process(item)
            results.append(result)
            pbar.update(1)
        
        return results
```

---

## Testing

### Unit Test Example

```python
import unittest
import torch

class TestMyNode(unittest.TestCase):
    def setUp(self):
        self.node = MyCustomNode()
    
    def test_basic_execution(self):
        input_tensor = torch.rand(1, 64, 64, 3)
        result = self.node.execute(input_tensor)
        
        self.assertEqual(result[0].shape, input_tensor.shape)
        self.assertTrue(torch.all(result[0] >= 0))
        self.assertTrue(torch.all(result[0] <= 1))
    
    def test_batch_processing(self):
        batch = torch.rand(4, 64, 64, 3)
        result = self.node.execute(batch)
        
        self.assertEqual(result[0].shape[0], 4)

if __name__ == "__main__":
    unittest.main()
```

---

## Common Issues and Solutions

### Issue: Shape Mismatch

```python
# Wrong: Assuming fixed batch size
def wrong(self, image):
    return image[0]  # Loses batch dimension

# Right: Preserve batch dimension
def right(self, image):
    return image  # Keep (B, H, W, C)
```

### Issue: Device Mismatch

```python
# Wrong: Mixing CPU and CUDA tensors
def wrong(self, tensor1, tensor2):
    return tensor1 + tensor2  # May fail if different devices

# Right: Ensure same device
def right(self, tensor1, tensor2):
    device = tensor1.device
    tensor2 = tensor2.to(device)
    return tensor1 + tensor2
```

### Issue: Value Range

```python
# Wrong: PIL expects 0-255, tensor is 0-1
pil_img = Image.fromarray(tensor.numpy())

# Right: Scale appropriately
pil_img = Image.fromarray((tensor.numpy() * 255).astype(np.uint8))
```
