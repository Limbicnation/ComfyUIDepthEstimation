# Import your existing ComfyUIDepthEstimationNode and the Depth Anything V2 classes

from .depth_estimation_node import ComfyUIDepthEstimationNode
from .depth_anything_v2_code import DepthAnything_V2, DownloadAndLoadDepthAnythingV2Model

NODE_CLASS_MAPPINGS = {
    # Existing ComfyUIDepthEstimationNode
    "ComfyUIDepthEstimationNode": ComfyUIDepthEstimationNode,

    # Depth Anything V2 related nodes
    "DepthAnything_V2": DepthAnything_V2,
    "DownloadAndLoadDepthAnythingV2Model": DownloadAndLoadDepthAnythingV2Model,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Display name for existing node
    "ComfyUIDepthEstimationNode": "Depth Estimation Node",

    # Display names for Depth Anything V2 nodes
    "DepthAnything_V2": "Depth Anything V2",
    "DownloadAndLoadDepthAnythingV2Model": "DownloadAndLoadDepthAnythingV2Model",
}
