## <img src="images/depth-estimation-icon.svg" width="32" height="32" alt="Depth Estimation Icon" style="vertical-align: middle"> ComfyUIDepthEstimation ComfyUIDepthEstimation

<div align="center">
  <img src="images/depth-estimation-icon.svg" width="150" height="150" alt="Depth Estimation Logo">
</div>

A custom depth estimation node for ComfyUI using Depth-Anything models to generate depth maps from images.

## Features
- Multiple model options:
  - Depth-Anything-Small
  - Depth-Anything-Base
  - Depth-Anything-Large
  - Depth-Anything-V2-Small
  - Depth-Anything-V2-Base
- Post-processing options:
  - Gaussian blur (adjustable radius)
  - Median filtering (configurable size)
  - Automatic contrast enhancement
  - Gamma correction

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ComfyUIDepthEstimation.git
   ```

2. Navigate to the repository directory:

   ```bash
   cd ComfyUIDepthEstimation
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

# Usage

<div align="center">
  <img src="images/depth-estimation-node.png" width="600" alt="Depth Estimation Node Preview">
</div>

## Node Parameters

Node Parameters

image: Input image (IMAGE type)
model_name: Select from available Depth-Anything models
blur_radius: Gaussian blur radius (0.0 - 10.0, default: 2.0)
median_size: Median filter size (3, 5, 7, 9, 11)
apply_auto_contrast: Enable automatic contrast enhancement
apply_gamma: Enable gamma correction

### Integrate with ComfyUI

1. Copy the `depth_estimation_node.py` file to your ComfyUI custom nodes directory.

2. Update ComfyUI configuration to include the custom node if necessary.

3. Restart ComfyUI to load the new custom node.

### Example Usage
1. Add the `Depth Estimation` node to your ComfyUI workflow
2. Connect an image source to the node's image input
3. Configure the parameters:
   - Select a model (e.g., "Depth-Anything-V2-Small")
   - Adjust blur_radius (0-10) for depth map smoothing
   - Choose median_size (3-11) for noise reduction
   - Toggle auto_contrast and gamma correction as needed
4. Connect the output to a Preview Image node or other image processing nodes

The node will process the input image and output a depth map that can be used in your ComfyUI workflow.

## License

This project is licensed under the Apache License.

---
