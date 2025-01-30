"""
ComfyUI Depth Estimation Node
A custom node for depth map estimation using Depth-Anything-V2-Small model.
"""

import os
import folder_paths

# Configure model paths
if not hasattr(folder_paths, "models_dir"):
    folder_paths.models_dir = os.path.join(folder_paths.base_path, "models")

# Register depth models path
DEPTH_DIR = "depth_anything"
models_path = os.path.join(folder_paths.models_dir, DEPTH_DIR)
os.makedirs(models_path, exist_ok=True)
folder_paths.folder_names_and_paths[DEPTH_DIR] = ([models_path], folder_paths.supported_pt_extensions)

from .depth_estimation_node import DepthEstimationNode

# Version info
__version__ = "1.0.0"

# Node class mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "DepthEstimationNode": DepthEstimationNode
}

# Display names for UI 
NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthEstimationNode": "Depth Estimation"
}

# Web extension info for ComfyUI
WEB_DIRECTORY = "./js"

# Module exports
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
    "WEB_DIRECTORY"
]