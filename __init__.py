"""
ComfyUI Depth Estimation Node
A custom node for depth map estimation using transformer models.
"""

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