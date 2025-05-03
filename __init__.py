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
__version__ = "1.0.0"

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
    "PIL": "9.1.0",  # Pillow is imported as PIL
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
    # Create a placeholder node that displays an error message
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
    
    logger.error(f"DepthEstimation Node disabled due to missing dependencies: {', '.join(missing_dependencies)}")
    logger.error(f"Please install with: pip install {' '.join(missing_dependencies)}")
else:
    # All dependencies are available, import the actual node
    try:
        # Try to import the improved node implementation first
        from .depth_estimation_node_improved import DepthEstimationNode, NodeDiagnostics
        logger.info("Using improved depth estimation node implementation")
        
        # Run diagnostics on startup
        try:
            if NodeDiagnostics:
                diag_info = NodeDiagnostics.get_system_info()
                logger.info(f"System info: Python {diag_info['python_version'].split()[0]}, "
                           f"Torch {diag_info['torch_version']}, "
                           f"CUDA: {diag_info['cuda_available']}")
                
                health = NodeDiagnostics.check_health()
                issues = [check for check in health if check["status"] == "fail"]
                if issues:
                    logger.warning(f"Found {len(issues)} potential issues: "
                                  f"{', '.join([issue['name'] for issue in issues])}")
        except Exception as e:
            logger.warning(f"Error running diagnostics: {e}")
        
    except ImportError as e:
        logger.warning(f"Could not import improved node: {e}. Falling back to original implementation.")
        # Fallback to original implementation
        try:
            from .depth_estimation_node import DepthEstimationNode
        except ImportError as e:
            logger.error(f"Failed to import main node implementation: {e}")
            
            # Create minimal placeholder if even the fallback fails
            class DepthEstimationNode:
                @classmethod
                def INPUT_TYPES(cls):
                    return {"required": {"image": ("IMAGE",)}}
                RETURN_TYPES = ("IMAGE",)
                FUNCTION = "estimate_depth"
                CATEGORY = "depth"
                
                def estimate_depth(self, image):
                    return (image,)  # Just pass through the image
    
    # Register the actual depth estimation node
    NODE_CLASS_MAPPINGS = {
        "DepthEstimationNode": DepthEstimationNode
    }
    
    NODE_DISPLAY_NAME_MAPPINGS = {
        "DepthEstimationNode": "Depth Estimation"
    }

# Module exports
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
    "WEB_DIRECTORY"
]