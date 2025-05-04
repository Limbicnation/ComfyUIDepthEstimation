"""
ComfyUI Depth Estimation Node
A custom node for depth map estimation using Depth-Anything models.
"""

import os
import logging
import importlib.util

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DepthEstimation")

# Version info
__version__ = "1.1.0"

# Node class mappings - will be populated based on dependency checks
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Web extension info for ComfyUI
WEB_DIRECTORY = "./js"

# Graceful dependency checking
required_dependencies = {
    "torch": "2.0.0",
    "transformers": "4.20.0",
    "numpy": "1.23.0",
    "PIL": "9.2.0",  # Pillow is imported as PIL
    "timm": "0.6.12",
    "huggingface_hub": "0.16.0"
}

missing_dependencies = []

# Check each dependency
for module_name, min_version in required_dependencies.items():
    try:
        if module_name == "PIL":
            # Special case for Pillow/PIL
            import PIL
            module_version = PIL.__version__
        else:
            module = __import__(module_name)
            module_version = getattr(module, "__version__", "unknown")
        
        logger.info(f"Found {module_name} version {module_version}")
    except ImportError:
        missing_dependencies.append(f"{module_name}>={min_version}")
        logger.warning(f"Missing required dependency: {module_name}>={min_version}")

if missing_dependencies:
    # Create placeholder node with dependency error
    class DependencyErrorNode:
        """Placeholder node that shows dependency installation instructions."""
        
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}
        
        RETURN_TYPES = ("STRING",)
        FUNCTION = "error_message"
        CATEGORY = "depth"
        
        def error_message(self):
            missing = ", ".join(missing_dependencies)
            message = f"Dependencies missing: {missing}. Please install with: pip install {' '.join(missing_dependencies)}"
            print(f"DepthEstimation Node Error: {message}")
            return (message,)
    
    # Register the error node instead of the real node
    NODE_CLASS_MAPPINGS = {
        "DepthEstimationNode": DependencyErrorNode
    }
    
    NODE_DISPLAY_NAME_MAPPINGS = {
        "DepthEstimationNode": "Depth Estimation (Missing Dependencies)"
    }
else:
    # All dependencies are available, try to import the actual node
    try:
        from .depth_estimation_node import DepthEstimationNode
        
        # Register the actual depth estimation node
        NODE_CLASS_MAPPINGS = {
            "DepthEstimationNode": DepthEstimationNode
        }
        
        NODE_DISPLAY_NAME_MAPPINGS = {
            "DepthEstimationNode": "Depth Estimation"
        }
    except Exception as e:
        # Capture any import errors that might occur with transformers
        logger.error(f"Error importing depth estimation node: {str(e)}")
        
        # Create a more specific error node
        class TransformersErrorNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}
            
            RETURN_TYPES = ("STRING",)
            FUNCTION = "error_message"
            CATEGORY = "depth"
            
            def error_message(self):
                if "Descriptors cannot be created directly" in str(e):
                    message = "Protobuf version conflict. Run: pip install protobuf==3.20.3"
                else:
                    message = f"Error loading depth estimation: {str(e)}"
                return (message,)
        
        NODE_CLASS_MAPPINGS = {
            "DepthEstimationNode": TransformersErrorNode
        }
        
        NODE_DISPLAY_NAME_MAPPINGS = {
            "DepthEstimationNode": "Depth Estimation (Error)"
        }

# Module exports
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
    "WEB_DIRECTORY"
]