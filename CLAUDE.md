# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI Depth Estimation Node is a custom node for ComfyUI that provides depth map generation from images using Depth-Anything models (V1, V2, and V3). The node integrates with ComfyUI's workflow system and provides configurable post-processing options. **Version 1.3.4+ adds Depth-Anything V3 support with camera pose estimation (extrinsics/intrinsics). Version 1.3.6 fixes import robustness for RunPod/PyTorch nightly environments.**

## Development Commands

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# For DA3 camera pose estimation (optional)
pip install depth_anything_3
```

### Testing the Node
The node is designed to run within ComfyUI. To test:
1. Copy/symlink this directory to `ComfyUI/custom_nodes/ComfyUIDepthEstimation`
2. Restart ComfyUI
3. Load the example workflow from `workflows/da3_camera_demo.json` (for DA3 features)

### Publishing
The project uses ComfyUI Registry for distribution:
```bash
# Project metadata is in pyproject.toml
# Publishing is configured in publish.yaml
```

## Code Architecture

### Entry Point: `__init__.py`
- Implements graceful dependency checking to avoid ComfyUI crashes
- Creates placeholder error nodes if dependencies are missing
- Registers the node with ComfyUI via `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
- Falls back to error nodes with helpful messages if imports fail
- **Important**: Variables in optional dependency loops must be initialized BEFORE try blocks to avoid scoping issues
- Configures logging via `logging.basicConfig()` - library modules should NOT call this (only `__init__.py`)

### Main Implementation: `depth_estimation_node.py`

#### Model System
The node supports multiple depth estimation models defined in `DEPTH_MODELS` dictionary:
- **Depth-Anything V1**: Small, Base, Large variants
- **Depth-Anything V2**: Small, Base variants
- **Depth-Anything V3**: Small, Base variants (Apache 2.0, with camera pose estimation)

Each model entry includes: HuggingFace path, VRAM requirements, direct download URL, model type, encoder type, and `supports_pose` flag.

#### DA3 Camera Pose Estimation (New in v1.3.4)
The node now supports camera extrinsics and intrinsics estimation for DA3 models:
- **DA3Prediction dataclass**: Structured output with depth, confidence, extrinsics, intrinsics, raw_depth
- **DA3ModelWrapper**: Wraps depth_anything_3 library with pose extraction
- **5-tuple output**: `(depth, confidence, extrinsics, intrinsics, camera_json)`

#### Main Node Class: `DepthEstimationNode`
**Input Parameters:**
- `image`: ComfyUI IMAGE tensor
- `model_name`: Dropdown of available models from DEPTH_MODELS
- `input_size`: Target size for inference (384-8192, default 1024)
- `blur_radius`: Gaussian blur radius (FLOAT, 0.0-100.0)
- `median_size`: Median filter size (INT, odd values only for PIL compatibility)
- `apply_auto_contrast`: Boolean for contrast enhancement
- `apply_gamma`: Boolean for gamma correction
- `force_reload`: Force model reload (optional)
- `force_cpu`: Force CPU processing (optional)
- `enable_camera_estimation`: Enable camera pose output for DA3 models (default True)
- `output_raw_depth`: Output raw metric depth instead of normalized (optional)

**Output Types:**
```python
RETURN_TYPES = ("IMAGE", "IMAGE", "CAMERA_EXTRINSICS", "CAMERA_INTRINSICS", "STRING")
RETURN_NAMES = ("depth", "confidence", "extrinsics", "intrinsics", "camera_json")
```

**Main Method:** `estimate_depth()`
- Validates input size and parameters
- Loads/reloads models with intelligent caching
- For DA3 models: extracts camera extrinsics [N, 3, 4] and intrinsics [N, 3, 3]
- Serializes camera data to comprehensive JSON with depth/confidence stats
- Returns 5-tuple with all outputs

#### Key Helper Methods
- `_serialize_camera_data()`: Converts camera tensors to structured JSON
- `_check_pose_support()`: Determines if model supports camera estimation
- `cleanup()`: Cleans up VRAM and manages model resources

### Integration with ComfyUI
- Uses `folder_paths` for ComfyUI path management
- Uses `comfy.model_management` for device selection and memory management
- Custom types: `CAMERA_EXTRINSICS`, `CAMERA_INTRINSICS` for type safety
- Node registered in "depth" category

## Camera JSON Output Format

When using DA3 models with camera estimation enabled:
```json
{
  "format_version": "1.0",
  "model": "Depth-Anything-V3-Base",
  "timestamp": 1234567890.123,
  "image_width": 1024,
  "image_height": 1024,
  "num_views": 1,
  "extrinsics": [{
    "matrix": [[...], [...], [...]],
    "format": "opencv_w2c",
    "rotation": [[...], [...], [...]],
    "translation": [x, y, z]
  }],
  "intrinsics": [{
    "matrix": [[...], [...], [...]],
    "fx": 1176.38,
    "fy": 1167.86,
    "cx": 252.0,
    "cy": 252.0
  }],
  "depth_stats": {"min": 0.68, "max": 1.37, "mean": 0.97, "std": 0.17},
  "confidence_stats": {"mean": 1.03, "std": 0.07, "min": 1.0, "max": 1.42}
}
```

## Important Implementation Details

### Model Loading Strategy
Three-tier fallback:
1. HuggingFace transformers pipeline (preferred)
2. Direct PyTorch model download from GitHub releases
3. Local cached models

### VRAM Management
- Models cached in `self.depth_estimator` to avoid reloading
- Explicit cleanup with `torch.cuda.empty_cache()` and `gc.collect()`
- VRAM requirements: V1/V2 (1.5-4GB), DA3 (~2-2.5GB)

### Image Format Conversions
- ComfyUI IMAGE tensors: BHWC format, 0-1 range
- PIL Images: RGB, 0-255 range
- Camera tensors: [N, 3, 4] for extrinsics, [N, 3, 3] for intrinsics

## ComfyUI Node Conventions

When modifying or extending this node:
- INPUT_TYPES must return a dict with "required" and optional "optional" keys
- Input types use ComfyUI type strings: "IMAGE", "INT", "FLOAT", "BOOLEAN", "STRING"
- Parameter constraints defined in dicts: `{"default": X, "min": Y, "max": Z, "step": W}`
- RETURN_TYPES is a tuple of ComfyUI type strings
- Custom types (CAMERA_EXTRINSICS, CAMERA_INTRINSICS) defined as string constants
- All nodes must be registered in `NODE_CLASS_MAPPINGS` in `__init__.py`

## Known Issues and Workarounds

1. **Protobuf errors**: Fixed by pinning to protobuf==3.20.3
2. **Model download failures**: Node includes retry logic and fallback URLs
3. **VRAM issues**: Use smaller models or enable `force_cpu` flag
4. **DA3 not available**: Node gracefully hides DA3 models if depth_anything_3 not installed
5. **Median filter**: PIL requires odd integers > 1; validation added
6. **PyTorch nightly builds**: v1.3.6 adds defensive import guards for edge cases on RunPod and similar environments

## Version History

- **v1.3.6** (2025-12-08): Fixed scoping bug in optional dependency checker, added defensive import guards for DA3, improved compatibility with PyTorch nightly builds
- **v1.3.5**: DA3 camera pose estimation feature release
- **v1.3.4**: Initial Depth-Anything V3 integration

## File Locations

- Example workflows: `workflows/da3_camera_demo.json`, `workflows/Depth_Map_Generator_V1.json`
- Icon: `icon.png` (referenced in pyproject.toml)
- Images: `images/` directory
