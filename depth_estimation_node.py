import os
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from controlnet_aux.util import custom_hf_download, DEPTH_ANYTHING_MODEL_NAME
from controlnet_aux.depth_anything.depth_anything.dpt import DPT_DINOv2
from controlnet_aux.depth_anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
import torch.nn.functional as F
from einops import repeat

def common_input_validate(input_image, output_type, **kwargs):
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)
    if input_image.ndim == 2:
        input_image = np.repeat(input_image[:, :, np.newaxis], 3, axis=2)
    if input_image.shape[2] == 1:
        input_image = np.repeat(input_image, 3, axis=2)
    if output_type is None:
        output_type = "np"
    return input_image, output_type

def resize_image_with_pad(image, resolution, interpolation=cv2.INTER_CUBIC):
    h, w = image.shape[:2]
    scale = resolution / min(h, w)
    nh, nw = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (nw, nh), interpolation=int(interpolation))
    pad_h = (resolution - nh) // 2
    pad_w = (resolution - nw) // 2
    image_padded = np.pad(image_resized, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
    def remove_pad(x):
        return x[pad_h:pad_h + nh, pad_w:pad_w + nw]
    return image_padded, remove_pad

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

DPT_CONFIGS = {
    "depth_anything_vitl14.pth": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024], "use_bn": False, "use_clstoken": False},
    "depth_anything_vitb14.pth": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768], "use_bn": False, "use_clstoken": False},
    "depth_anything_vits14.pth": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384], "use_bn": False, "use_clstoken": False}
}

class DepthAnythingDetector:
    def __init__(self, model):
        self.model = model
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=DEPTH_ANYTHING_MODEL_NAME, filename="depth_anything_vitl14.pth"):
        model_path = custom_hf_download(pretrained_model_or_path, filename, subfolder="checkpoints", repo_type="space")
        model = DPT_DINOv2(**DPT_CONFIGS[filename], localhub=True)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return cls(model)

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self
    
    def __call__(self, input_image, detect_resolution=512, output_type=None, upscale_method=cv2.INTER_CUBIC, **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        t, remove_pad = resize_image_with_pad(np.zeros_like(input_image), detect_resolution, upscale_method)
        t = remove_pad(t)

        h, w = t.shape[:2]
        h, w = int(h), int(w)
        image = transform({'image': input_image / 255.})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            depth = self.model(image)
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        detected_map = repeat(depth, "h w -> h w 3").cpu().numpy().astype(np.uint8)
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map

class DepthEstimationNode:
    def __init__(self):
        self.blur_radius = 2.0
        self.median_size = self.ensure_odd(5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = self.initialize_detector()

    def initialize_detector(self):
        """Initialize the depth estimation detector with detailed logging."""
        try:
            print("Initializing depth estimation detector...")
            detector = DepthAnythingDetector.from_pretrained()
            detector.to(self.device)
            print("Detector initialized successfully.")
            return detector
        except Exception as e:
            print(f"Error initializing detector: {e}")
            return None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("FLOAT", {"default": 2.0}),
                "median_size": ("INT", {"default": 5}),
                "device": (["cpu", "gpu"], {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_image",)
    FUNCTION = "execute"
    CATEGORY = "Depth Estimation"

    def ensure_odd(self, value):
        """Ensure the value is an odd integer."""
        value = int(value)
        return value if value % 2 == 1 else value + 1

    def gamma_correction(self, img, gamma=1.0):
        """Apply gamma correction to the image."""
        inv_gamma = 1.0 / gamma
        table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        return Image.fromarray(np.array(img).astype(np.uint8)).point(lambda i: table[i])

    def process_image(self, image, blur_radius, median_size):
        # Ensure detector is initialized
        if self.detector is None:
            raise RuntimeError("Pipeline not initialized properly.")

        # Convert image to numpy array
        np_image = np.array(image)

        # Ensure the data type is uint8 and has 3 channels
        if np_image.dtype != np.uint8:
            np_image = np_image.astype(np.uint8)
        if np_image.shape[2] == 4:
            np_image = np_image[:, :, :3]

        # Run the detector
        detected_map = self.detector(input_image=np_image, output_type="pil")

        # Apply a median filter to reduce noise
        detected_map = detected_map.filter(ImageFilter.MedianFilter(size=self.ensure_odd(median_size)))

        # Enhanced edge detection with more feathering
        edges = detected_map.filter(ImageFilter.FIND_EDGES)
        edges = edges.filter(ImageFilter.GaussianBlur(radius=2 * blur_radius))
        edges = edges.point(lambda x: 255 if x > 20 else 0)  # Adjusted threshold

        # Create a mask from the edges
        mask = edges.convert("L")

        # Blur only the edges using the mask
        blurred_edges = detected_map.filter(ImageFilter.GaussianBlur(radius=blur_radius * 2))

        # Combine the blurred edges with the original depth image using the mask
        combined_image = Image.composite(blurred_edges, detected_map, mask)

        # Apply auto gamma correction with a lower gamma to darken the image
        gamma_corrected_image = self.gamma_correction(combined_image, gamma=0.7)

        # Apply auto contrast
        final_image = ImageOps.autocontrast(gamma_corrected_image)

        # Additional post-processing: Sharpen the final image
        final_image = final_image.filter(ImageFilter.SHARPEN)

        return final_image

    def execute(self, image, blur_radius=2.0, median_size=5, device="cpu"):
        # Check if the input is a Tensor or NumPy array; if so, convert it to a PIL Image
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(image, np.ndarray):
            if image.ndim == 4 and image.shape[0] == 1:
                image = image[0]  # Remove the batch dimension if present
            image = image.astype(np.uint8)
            if image.shape[2] == 4:
                image = image[:, :, :3]
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise ValueError("Input should be a PIL Image or a Tensor convertible to a PIL Image")

        # Update device
        self.device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")

        final_image = self.process_image(image, blur_radius, median_size)
        return (final_image, )

NODE_CLASS_MAPPINGS = {
    "DepthEstimationNode": DepthEstimationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthEstimationNode": "Depth Estimation Node"
}
