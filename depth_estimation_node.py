import os
import numpy as np
import torch
from transformers import pipeline
from PIL import Image, ImageFilter, ImageOps
from comfy.nodes import Node, register_node

def ensure_odd(value):
    """Ensure the value is an odd integer."""
    value = int(value)
    return value if value % 2 == 1 else value + 1

def convert_path(path):
    """Convert path for compatibility between Windows and WSL."""
    if os.name == 'nt':  # If running on Windows
        return path.replace('\\', '/')
    return path

def gamma_correction(img, gamma=1.0):
    """Apply gamma correction to the image."""
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return Image.fromarray(np.array(img).astype(np.uint8)).point(lambda i: table[i])

def auto_gamma_correction(image):
    """Automatically adjust gamma correction for the image."""
    image_array = np.array(image).astype(np.float32) / 255.0
    mean_luminance = np.mean(image_array)
    gamma = np.log(0.5) / np.log(mean_luminance)
    return gamma_correction(image, gamma=gamma)

def auto_contrast(image):
    """Apply automatic contrast adjustment to the image."""
    return ImageOps.autocontrast(image)

class DepthEstimationNode(Node):
    def __init__(self, blur_radius=2.0, median_size=5, device="cpu"):
        super().__init__()
        self.blur_radius = blur_radius
        self.median_size = ensure_odd(median_size)
        self.device = 0 if device == "gpu" and torch.cuda.is_available() else -1
        self.pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf", device=self.device)

    def process_image(self, image):
        if self.device == 0:
            image = image.convert("RGB")  # Ensure image is in RGB format
            inputs = self.pipe.feature_extractor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.pipe.model(**inputs)
            result = self.pipe.post_process(outputs, (image.height, image.width))
        else:
            result = self.pipe(image)

        # Convert depth data to a NumPy array if not already one
        depth_data = np.array(result["depth"])

        # Normalize and convert to uint8
        depth_normalized = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min() + 1e-8)  # Avoid zero division
        depth_uint8 = (255 * depth_normalized).astype(np.uint8)

        # Create an image from the processed depth data
        depth_image = Image.fromarray(depth_uint8)

        # Apply a median filter to reduce noise
        depth_image = depth_image.filter(ImageFilter.MedianFilter(size=self.median_size))

        # Enhanced edge detection with more feathering
        edges = depth_image.filter(ImageFilter.FIND_EDGES)
        edges = edges.filter(ImageFilter.GaussianBlur(radius=2 * self.blur_radius))
        edges = edges.point(lambda x: 255 if x > 20 else 0)  # Adjusted threshold

        # Create a mask from the edges
        mask = edges.convert("L")

        # Blur only the edges using the mask
        blurred_edges = depth_image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius * 2))

        # Combine the blurred edges with the original depth image using the mask
        combined_image = Image.composite(blurred_edges, depth_image, mask)

        # Apply auto gamma correction with a lower gamma to darken the image
        gamma_corrected_image = gamma_correction(combined_image, gamma=0.7)

        # Apply auto contrast
        final_image = auto_contrast(gamma_corrected_image)

        # Additional post-processing: Sharpen the final image
        final_image = final_image.filter(ImageFilter.SHARPEN)

        return final_image

    def forward(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The input image path does not exist: {image_path}")

        image = Image.open(image_path)
        return self.process_image(image)

@register_node
class ComfyUIDepthEstimationNode(DepthEstimationNode):
    def __init__(self):
        super().__init__(blur_radius=2.0, median_size=5, device="cpu")

    def execute(self, image_path: str, output_path: str):
        final_image = self.forward(image_path)
        final_image.save(output_path)
        print(f"Processed and saved: {output_path}")
