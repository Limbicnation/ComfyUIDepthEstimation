from .depth_estimation_node import DepthEstimationNode

NODE_CLASS_MAPPINGS = {
    "DepthEstimationNode": DepthEstimationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthEstimationNode": "Depth Estimation Node"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
