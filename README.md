---

# 📐 ComfyUIDepthEstimation

A custom depth estimation node for ComfyUI using transformer models. It integrates depth estimation with automatic gamma correction, contrast adjustment, and edge detection, based on the [TransformDepth](https://github.com/Limbicnation/TransformDepth) repository.

## Features

- Depth estimation using a transformer model
- Automatic gamma correction
- Automatic contrast adjustment
- Edge detection and blurring

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

## Usage

### Integrate with ComfyUI

1. Copy the `depth_estimation_node.py` file to your ComfyUI custom nodes directory.

2. Update ComfyUI configuration to include the custom node if necessary.

3. Restart ComfyUI to load the new custom node.

### Example Usage

1. Add the `ComfyUIDepthEstimationNode` to your pipeline in ComfyUI.

2. Configure the node with the necessary parameters:
   - `image_path`: Path to the input image.
   - `output_path`: Path to save the processed image.
   - `blur_radius`: Radius for Gaussian Blur.
   - `median_size`: Size for Median Filter.
   - `device`: Device to use for inference (cpu or gpu).

## License

This project is licensed under the Apache License.

---
