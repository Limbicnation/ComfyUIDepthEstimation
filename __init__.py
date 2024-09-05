from .your_node_file import ComfyUIDepthEstimationNode

NODE_CLASS_MAPPINGS = {
    "ComfyUIDepthEstimationNode": ComfyUIDepthEstimationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUIDepthEstimationNode": "Depth Estimation Node"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
