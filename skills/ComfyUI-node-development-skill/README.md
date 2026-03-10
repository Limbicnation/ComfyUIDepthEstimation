# ComfyUI Node Development Skill

A comprehensive skill for developing ComfyUI custom nodes following production best practices.

## Overview

This skill provides guidance for creating ComfyUI custom nodes, from basic structure to advanced patterns like memory management, batch processing, and model integration.

## Installation

### Local (Repository Only)
```bash
cp -r ComfyUI-node-development-skill ~/.claude/skills/
```

### Global Installation
```bash
ln -s "$(pwd)/ComfyUI-node-development-skill" ~/.claude/skills/comfyui-node-development
```

## Usage

This skill automatically activates when you mention:
- "create ComfyUI node"
- "build custom node"
- "ComfyUI development"
- "node for ComfyUI"
- "custom ComfyUI"

## Quick Start

### Basic Node Template

```python
class MyCustomNode:
    """Description of what this node does."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": ("INPUT_TYPE", {"default": value}),
            }
        }
    
    RETURN_TYPES = ("OUTPUT_TYPE",)
    FUNCTION = "execute"
    CATEGORY = "custom/category"
    
    def execute(self, input_name):
        # Your logic here
        return (output,)
```

### Register Your Node

In `__init__.py`:

```python
from .my_module import MyCustomNode

NODE_CLASS_MAPPINGS = {
    "MyCustomNode": MyCustomNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyCustomNode": "My Custom Node",
}
```

## Features

- ✅ Node architecture and class structure
- ✅ Input/output type system
- ✅ Widget configurations
- ✅ Model loading patterns
- ✅ Memory management
- ✅ Batch processing
- ✅ Error handling
- ✅ Testing patterns

## Examples

See the `examples/` directory for working node implementations:
- `basic_image_processor.py` - Simple image manipulation
- `model_loader.py` - Custom model loading
- `batch_processor.py` - Efficient batch operations

## References

- [ComfyUI Documentation](https://docs.comfy.org/)
- [Custom Nodes Guide](https://docs.comfy.org/essentials/custom_nodes/)
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
