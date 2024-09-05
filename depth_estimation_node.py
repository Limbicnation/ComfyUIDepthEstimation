import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import argparse
from contextlib import nullcontext
from PIL import Image, ImageFilter, ImageOps
import cv2

import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file
import folder_paths

from .depth_anything_v2.dpt import DepthAnythingV2
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device


def gamma_correction(img: Image.Image, gamma: float) -> Image.Image:
    """Apply gamma correction to the image."""
    if gamma <= 0:
        raise ValueError("Gamma value must be greater than zero.")
    
    inv_gamma = 1.0 / gamma
    max_pixel_value = 65535 if img.mode == "I;16" else 255
    table = [(i / max_pixel_value) ** inv_gamma * max_pixel_value for i in range(max_pixel_value + 1)]
    table = np.array(table, dtype=np.uint16 if max_pixel_value == 65535 else np.uint8)
    
    if img.mode == "I;16":
        img = img.point(lambda i: table[i], 'I;16')
    else:
        img = img.point(lambda i: table[i])
    
    return img


def apply_median_filter(image: Image.Image, size: int) -> Image.Image:
    """Apply a median filter to the image."""
    return image.filter(ImageFilter.MedianFilter(size=size))


def detect_edges(image: Image.Image) -> Image.Image:
    """Detect edges in the image."""
    return image.filter(ImageFilter.CONTOUR)


class DepthAnything_V2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "da_model": ("DAMODEL", ),
            "images": ("IMAGE", ),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("image",)
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV2"
    DESCRIPTION = """
https://depth-anything-v2.github.io
"""

    def process(self, da_model, images):
        device = mm.get_torch_device()
        model = da_model['model']
        dtype=da_model['dtype']
        
        B, H, W, C = images.shape

        images = images.permute(0, 3, 1, 2)

        orig_H, orig_W = H, W
        if W % 14 != 0:
            W = W - (W % 14)
        if H % 14 != 0:
            H = H - (H % 14)
        if orig_H % 14 != 0 or orig_W % 14 != 0:
            images = F.interpolate(images, size=(H, W), mode="bilinear")
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalized_images = normalize(images)
        pbar = ProgressBar(B)
        out = []
        model.to(device)
        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            for img in normalized_images:
                depth = model(img.unsqueeze(0).to(device))
                depth = (depth - depth.min()) / (depth.max() - depth.min())
                out.append(depth.cpu())
                pbar.update(1)
            depth_out = torch.cat(out, dim=0)
            depth_out = depth_out.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()
        
        return (depth_out,)


class DownloadAndLoadDepthAnythingV2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                        'depth_anything_v2_vits_fp16.safetensors',
                        'depth_anything_v2_vits_fp32.safetensors',
                        'depth_anything_v2_vitb_fp16.safetensors',
                        'depth_anything_v2_vitb_fp32.safetensors',
                        'depth_anything_v2_vitl_fp16.safetensors',
                        'depth_anything_v2_vitl_fp32.safetensors',
                        'depth_anything_v2_metric_hypersim_vitl_fp32.safetensors',
                        'depth_anything_v2_metric_vkitti_vitl_fp32.safetensors'
                    ],
                    {
                    "default": 'depth_anything_v2_vitl_fp32.safetensors'
                    }),
            },
        }

    RETURN_TYPES = ("DAMODEL",)
    RETURN_NAMES = ("da_v2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "DepthAnythingV2"

    def loadmodel(self, model):
        device = mm.get_torch_device()
        dtype = torch.float16 if "fp16" in model else torch.float32
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        download_path = os.path.join(folder_paths.models_dir, "depthanything")
        model_path = os.path.join(download_path, model)

        if not os.path.exists(model_path):
            print(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Kijai/DepthAnythingV2-safetensors", 
                              allow_patterns=[f"*{model}*"],
                              local_dir=download_path, 
                              local_dir_use_symlinks=False)

        print(f"Loading model from: {model_path}")
        encoder = 'vitl' if 'vitl' in model else 'vitb' if 'vitb' in model else 'vits'

        with (init_empty_weights() if True else nullcontext()):
            self.model = DepthAnythingV2(**model_configs[encoder])

        state_dict = load_torch_file(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        return {"model": self.model, "dtype": dtype}


def process_image(image_path, output_path, blur_radius, median_size, flag, no_post_processing, apply_gamma, gamma_value, model):
    image = Image.open(image_path)
    depth_output = model.process(image)
    
    if apply_gamma:
        depth_output = gamma_correction(depth_output, gamma_value)

    if not no_post_processing:
        depth_output = apply_median_filter(depth_output, median_size)
        depth_output = depth_output.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        edges = detect_edges(depth_output)
        combined = Image.blend(depth_output, edges, alpha=0.3)
        combined.save(output_path)
    else:
        depth_output.save(output_path)
    
    print(f"Processed and saved: {output_path}")


def main():
    """Main function to parse arguments and process images."""
    parser = argparse.ArgumentParser(description="Process images for depth estimation.")
    parser.add_argument("--single", type=str, help="Path to a single image file to process.")
    parser.add_argument("--batch", type=str, help="Path to directory of images to process in batch.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed images.")
    parser.add_argument("--blur_radius", type=float, default=2.0, help="Radius for Gaussian Blur. Default is 2.0. Can accept float values.")
    parser.add_argument("--median_size", type=int, default=5, help="Size for Median Filter. Default is 5. Must be an odd integer.")
    parser.add_argument("--depth-anything-v2-small", action='store_true', help="Flag to use Depth-Anything-V2-Small model.")
    parser.add_argument("--flag", action='store_true', help="A flag to trigger additional processing options.")
    parser.add_argument("--no-post-processing", action='store_true', help="Disable post-processing effects.")
    parser.add_argument("--apply-gamma", action='store_true', help="Apply gamma correction to the output.")
    parser.add_argument("--gamma-value", type=float, default=1.0, help="Gamma value for correction. Default is 1.0 (no correction).")
    args = parser.parse_args()

    # Load the Depth Anything V2 model
    model_loader = DownloadAndLoadDepthAnythingV2Model()
    model = model_loader.loadmodel('depth_anything_v2_vitl_fp32.safetensors')  # Adjust to use the correct flag

    # Process a single image or batch
    if args.single:
        output_path = os.path.join(args.output, 'depth-' + os.path.basename(args.single))
        process_image(args.single, output_path, args.blur_radius, args.median_size, args.flag, args.no_post_processing, args.apply_gamma, args.gamma_value, model)
    elif args.batch:
        for filename in os.listdir(args.batch):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(args.batch, filename)
                output_path = os.path.join(args.output, 'depth-' + filename)
                process_image(image_path, output_path, args.blur_radius, args.median_size, args.flag, args.no_post_processing, args.apply_gamma, args.gamma_value, model)
    else:
        print("Please specify either --single <image_path> or --batch <directory_path> to process images.")


if __name__ == "__main__":
    main()
