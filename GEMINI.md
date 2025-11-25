# ComfyUI Depth Estimation Node

## Project Overview
This project is a custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that generates depth maps from input images. It utilizes state-of-the-art monocular depth estimation models, specifically **Depth-Anything** (V1 and V2) and **MiDaS**.

The node is designed to be robust, offering multiple fallback mechanisms for model loading (Hugging Face `transformers` pipeline, direct `timm`/`torchvision` implementation, and manual weight downloading). It also includes post-processing features like Gaussian blur, median filtering, auto-contrast, and gamma correction.

## Key Files

- **`depth_estimation_node.py`**: The core implementation file. It contains:
    - `DepthEstimationNode`: The main ComfyUI node class defining inputs (`image`, `model_name`, parameters) and output (`IMAGE`).
    - `DepthAnythingV2`: A direct PyTorch implementation of the V2 architecture (using `timm` backbones like ViT).
    - `MiDaSWrapper`: A wrapper for loading and running MiDaS models as a fallback.
    - Model loading logic: Complex logic to handle caching, downloading from multiple sources, and falling back between loading methods (Pipeline -> Direct -> MiDaS).
    - Image processing: helper methods to convert between ComfyUI's Tensor/Numpy formats and PIL, including resizing and normalization.

- **`__init__.py`**: The entry point for ComfyUI.
    - Performs dependency checks (`torch`, `transformers`, `timm`, etc.).
    - Registers the node (`NODE_CLASS_MAPPINGS`).
    - Provides a fallback "Error Node" if dependencies are missing or if there are `protobuf` conflicts.

- **`requirements.txt`**: Lists Python dependencies. Key libraries include `transformers`, `timm`, `huggingface-hub`, `protobuf==3.20.3` (pinned for compatibility), and `tokenizers`.

- **`pyproject.toml`**: Project metadata and build configuration.

## Architecture & Logic

1.  **Model Loading Strategy**:
    - The node attempts to load models in a prioritized order:
        1.  **Transformers Pipeline**: Uses `pipeline("depth-estimation")` from Hugging Face.
        2.  **Direct Implementation**: Uses the local `DepthAnythingV2` class or `MiDaSWrapper` if the pipeline fails (e.g., due to auth or network issues).
    - Models are cached in `ComfyUI/models/depth_anything` (or similar paths defined in `folder_paths`).

2.  **Image Processing Pipeline**:
    - **Input**: Accepts ComfyUI `IMAGE` (Tensor [B, H, W, C]).
    - **Preprocessing**: Converts to PIL, resizes to target `input_size` (ensuring dimensions are multiples of 32).
    - **Inference**: Runs the model to get a depth map.
    - **Post-processing**:
        - **Blur**: Gaussian blur to smooth noise.
        - **Median Filter**: Removes salt-and-pepper noise.
        - **Normalization**: Normalizes depth values to 0-1.
        - **Enhancement**: Optional Auto-Contrast and Gamma correction.
    - **Output**: Converts back to Tensor for ComfyUI.

## Setup & Usage

### Installation
This node is typically installed into the `custom_nodes` directory of a ComfyUI instance.

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Limbicnation/ComfyUIDepthEstimation.git
cd ComfyUIDepthEstimation
pip install -r requirements.txt
```

### Dependencies
Critical dependencies include:
- `transformers` (>=4.20.0)
- `timm` (>=0.6.12)
- `protobuf` (==3.20.3) - *Important for preventing conflicts with some transformers versions.*

### Usage in ComfyUI
- **Node Name**: `DepthEstimationNode` (Display: "Depth Estimation")
- **Inputs**:
    - `image`: Source image.
    - `model_name`: Selector for models (e.g., "Depth-Anything-V2-Small").
    - `input_size`: Resolution for inference (default 1024).
    - `blur_radius`, `median_size`: Smoothing parameters.
- `input_size`: Resolution for inference (default 1024).
    - `blur_radius`, `median_size`: Smoothing parameters.
    - `force_cpu`: Option to offload VRAM.

## Recent Updates (DA3 Integration)
- **Depth Anything V3 Support**: Added `Depth-Anything-V3-Small` and `Depth-Anything-V3-Base` models.
- **Optional Dependency**: DA3 models require the `depth_anything_3` package. The node checks for its presence and gracefully hides DA3 models if missing.
- **Video Processing**: Native support for video processing via batch inputs.
- **Architecture**: Introduced `DA3ModelWrapper` to adapt the DA3 API (which uses `inference()`) to the node's expected interface.
- **Version Checking**: Robust version comparison for optional dependencies to ensure compatibility.
