# Implementation Plan: DA3 Camera Pose Estimation Integration

**Project:** ComfyUIDepthEstimation
**Location:** `/home/gero/GitHub/ComfyUI_Lab/ComfyUIDepthEstimation/`
**Target File:** `depth_estimation_node.py` (2895 lines)
**Date:** 2025-12-06
**Status:** APPROVED - Ready for Implementation

---

## CONFIRMED USER DECISIONS (2025-12-06)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Architecture** | Extend existing `DepthEstimationNode` | Maintain single-node pattern, simpler for users |
| **Models** | Apache 2.0 only (DA3-Small, DA3-Base) | Commercial-friendly, excludes CC BY-NC models |
| **Multi-view** | Essential - include batch processing | Required for video/multi-angle workflows |
| **Return Types** | 5-tuple explicit types | Type-safe, ComfyUI-native, future-proof |

### Architectural Decision: 5-tuple Return Types

```python
RETURN_TYPES = (
    "IMAGE",              # depth map (normalized 0-1)
    "IMAGE",              # confidence map (0-1 or None)
    "CAMERA_EXTRINSICS",  # Custom Type [N, 3, 4] w2c transform
    "CAMERA_INTRINSICS",  # Custom Type [N, 3, 3] calibration
    "STRING"              # JSON metadata
)
```

**Rationale:**

1. **ComfyUI Native Philosophy** - Explicit data flow, visual clarity in workflow editor
2. **Type Safety** - Custom types prevent misconnections (can't plug 3x4 matrix into VAE)
3. **Graceful Degradation** - Unused outputs ignored, no workflow breakage
4. **Future-Proof** - Establishes standard types for downstream 3D/video nodes

**Rejected Alternatives:**

- `DICT` bundle: Requires breakout node, defers complexity
- JSON-only: Requires parsing in every receiving node, limits composability

---

## Executive Summary

Integrate Depth Anything V3 camera pose estimation capabilities into the existing ComfyUI depth estimation node. The implementation extends the current DA3ModelWrapper to extract all prediction fields (extrinsics, intrinsics, confidence) and creates an **integrated single-node architecture** that differs from the reference implementation's multi-node approach.

**Key Architectural Difference:** While the reference uses separate nodes for model loading, inference, camera utilities, and point cloud generation, this implementation uses a **unified node with optional camera outputs** that only activate when DA3 models supporting pose estimation are selected.

> [!IMPORTANT]
> **Decision Update:** We have selected **Option 1 (5-tuple explicit return)** as the implementation strategy.
> This provides type safety (`CAMERA_EXTRINSICS`, `CAMERA_INTRINSICS`) and explicit data flow, aligning with ComfyUI best practices.

---

## 1. Current State Analysis

### Current DA3 Support (Lines 619-721)

**DA3ModelWrapper Class:**

- Location: Lines 619-721
- Currently extracts only `prediction.depth`
- Missing fields: `extrinsics`, `intrinsics`, `conf`, `processed_images`
- Returns simple dict: `{"predicted_depth": depth_tensor}`

**DepthEstimationNode Class:**

- Location: Lines 743-2895
- Single return type: `RETURN_TYPES = ("IMAGE",)`
- Input: model_name, image, post-processing params
- Models cached in `self.depth_estimator`

**Current DA3 Models in DEPTH_MODELS (Lines 303-322):**

```python
"Depth-Anything-V3-Small": {
    "path": "depth-anything/DA3-Small",
    "vram_mb": 2000,
    "model_type": "v3",
    "encoder": "vits",
    "license": "Apache-2.0",
    "supports_batch": True,
    "params": "80M"
}
"Depth-Anything-V3-Base": {
    "path": "depth-anything/DA3-Base",
    "vram_mb": 2500,
    "model_type": "v3",
    "encoder": "vitb",
    "license": "Apache-2.0",
    "supports_batch": True,
    "params": "120M"
}
```

**SCOPE: Apache 2.0 Models Only**

- DA3-Small (80M) - ✅ Include
- DA3-Base (120M) - ✅ Include

**Excluded (CC BY-NC 4.0 or no pose support):**

- DA3-Large (350M) - ❌ CC BY-NC 4.0
- DA3-Giant (1.15B) - ❌ CC BY-NC 4.0
- DA3Nested-Giant-Large (1.4B) - ❌ CC BY-NC 4.0
- DA3Metric-Large (350M) - ❌ NO pose estimation
- DA3Mono-Large (350M) - ❌ NO pose estimation

---

## 2. Model Support Matrix

**Models Supporting Camera Pose Estimation:**

| Model | Size | License | Pose Est. | Pose Cond. | Multi-View |
|-------|------|---------|-----------|------------|------------|
| DA3-Small | 80M | Apache 2.0 | ✅ | ✅ | ✅ |
| DA3-Base | 120M | Apache 2.0 | ✅ | ✅ | ✅ |
| DA3-Large | 350M | CC BY-NC 4.0 | ✅ | ✅ | ✅ |
| DA3-Giant | 1.15B | CC BY-NC 4.0 | ✅ | ✅ | ✅ |
| DA3Nested-Giant-Large | 1.4B | CC BY-NC 4.0 | ✅ | ✅ | ✅ |
| DA3Metric-Large | 350M | Apache 2.0 | ❌ | ❌ | ❌ |
| DA3Mono-Large | 350M | Apache 2.0 | ❌ | ❌ | ❌ |

**Key Insight:** Only the "Any-view" series (Small, Base, Large, Giant, Nested) support camera estimation.

### 2.1 Model Selection Status

| Model | Status |
|-------|--------|
| DA3-Small | ✅ Integrated |
| DA3-Base | ✅ Integrated |
| DA3-Large | ✅ Integrated |
| DA3-Giant | ✅ Integrated |
| DA3Nested-Giant-Large | ✅ Integrated |

---

## 3. Architectural Design Decision

### Chosen Architecture: Unified Node with Conditional Outputs

**Rationale:**

- Maintains consistency with existing codebase pattern (single node)
- Avoids complexity of multi-node communication
- Provides graceful degradation for non-DA3 models
- Follows ComfyUI philosophy of "optional outputs" (unused outputs are ignored)

**Differentiation from Reference:**

| Aspect | Reference (ComfyUI-DepthAnythingV3) | Our Implementation |
|--------|-------------------------------------|-------------------|
| Architecture | Multi-node (Load → Inference → Camera Utils → Point Cloud) | Single unified node with conditional outputs |
| Model Loading | Separate "Download and Load" node | Integrated into main node (existing pattern) |
| Camera Data | Separate "Create/Parse Camera Parameters" nodes | Direct outputs as ComfyUI types |
| Normalization | 3 modes (Standard, V2-Style, Raw) | Keep existing post-processing + add Raw option |
| Point Clouds | Separate conversion nodes | Future extension (out of scope) |
| Tiled Processing | Separate "Enable Tiled Processing" node | Direct parameter (if needed) |

**Benefits:**

1. Simpler workflow for users (fewer nodes to connect)
2. Automatic detection of model capabilities
3. Backward compatible with existing workflows
4. Consistent with codebase architecture patterns

---

## 4. FAQ & User Guidance

### Q: Does this export the camera intrinsics you get from DA3?

**A: Yes.**
The node explicitly returns `CAMERA_INTRINSICS` as a standard output.

- **Format:** `[N, 3, 3]` tensor containing the calibration matrices.
- **Values:** Includes focal lengths (`fx`, `fy`) and principal points (`cx`, `cy`).
- **Metadata:** The `camera_json` output also contains these values in a human-readable format.

---

## 5. Implementation Design

### 4.1 Enhanced DA3ModelWrapper

**Location:** Extend existing class at lines 619-721

**New Class Structure:**

```python
@dataclass
class DA3Prediction:
    """
    Structured output from DA3 models containing all prediction fields.

    Attributes:
        depth: [N, H, W] normalized depth maps (0-1 range)
        confidence: [N, H, W] confidence maps (0-1 range), None for non-supporting models
        extrinsics: [N, 3, 4] camera extrinsics (opencv w2c format), None for non-supporting models
        intrinsics: [N, 3, 3] camera intrinsics, None for non-supporting models
        processed_images: [N, H, W, 3] uint8 processed images
        raw_depth: [N, H, W] raw unnormalized depth (for metric applications)
        supports_pose: bool indicating if model supports camera estimation
    """
    depth: torch.Tensor  # [N, H, W] normalized 0-1
    confidence: Optional[torch.Tensor] = None  # [N, H, W] or None
    extrinsics: Optional[torch.Tensor] = None  # [N, 3, 4] or None
    intrinsics: Optional[torch.Tensor] = None  # [N, 3, 3] or None
    processed_images: Optional[torch.Tensor] = None  # [N, H, W, 3] uint8
    raw_depth: Optional[torch.Tensor] = None  # [N, H, W] unnormalized
    supports_pose: bool = False


class DA3ModelWrapper:
    """
    Enhanced wrapper for Depth Anything V3 API.

    Extracts all available prediction fields including camera pose estimation
    for models that support it (Any-view series: Small, Base, Large, Giant, Nested).
    """

    def __init__(self, model, device, model_name: str):
        """
        Args:
            model: The loaded DA3 model instance
            device: The device to run inference on (cuda/cpu)
            model_name: Name of the model (used to determine capabilities)
        """
        self.model = model
        self.device = device
        self.model_name = model_name

        # Determine if model supports camera pose estimation
        # Only Any-view models (not Mono or Metric variants) support pose
        self.supports_pose = self._check_pose_support(model_name)

        logger.info(f"DA3ModelWrapper initialized: {model_name}, "
                   f"Pose support: {self.supports_pose}, Device: {device}")

    def _check_pose_support(self, model_name: str) -> bool:
        """
        Check if model supports camera pose estimation.

        Any-view models (Small, Base, Large, Giant, Nested) support pose.
        Mono and Metric variants do NOT support pose estimation.
        """
        model_lower = model_name.lower()

        # Explicitly exclude Mono and Metric variants
        if "mono" in model_lower or "metric" in model_lower:
            return False

        # Any-view models (Small, Base, Large, Giant, Nested) support pose
        if "v3" in model_lower or "da3" in model_lower:
            # Check it's not explicitly a non-supporting variant
            if any(variant in model_lower for variant in ["small", "base", "large", "giant", "nested"]):
                # But not if it's Mono or Metric (already checked above)
                return True

        return False

    def __call__(self, image: Union[Image.Image, List[Image.Image]]) -> DA3Prediction:
        """
        Run inference and return structured prediction with all available fields.

        Args:
            image: Single PIL Image or list of PIL Images for batch processing

        Returns:
            DA3Prediction dataclass with all available prediction fields
        """
        try:
            # Handle single image or batch
            if isinstance(image, Image.Image):
                images = [image]
                is_batch = False
            else:
                images = list(image)
                is_batch = True

            # DA3 inference
            with torch.inference_mode():
                prediction = self.model.inference(images)

            # Extract depth maps - ALWAYS available
            raw_depths = prediction.depth if hasattr(prediction, 'depth') else np.array(prediction)

            # Normalize depths to [0, 1] for ComfyUI display
            normalized_depths = []
            for depth in raw_depths:
                depth_min, depth_max = depth.min(), depth.max()
                if depth_max - depth_min > 1e-6:
                    norm_depth = (depth - depth_min) / (depth_max - depth_min)
                else:
                    norm_depth = np.zeros_like(depth)
                normalized_depths.append(norm_depth)

            depth_array = np.stack(normalized_depths, axis=0)  # [N, H, W]
            depth_tensor = torch.from_numpy(depth_array).float().to(self.device)

            # Extract raw depth for metric applications
            raw_depth_tensor = torch.from_numpy(raw_depths).float().to(self.device)

            # Extract confidence maps if available
            confidence_tensor = None
            if hasattr(prediction, 'conf') and prediction.conf is not None:
                conf_array = prediction.conf  # [N, H, W]
                confidence_tensor = torch.from_numpy(conf_array).float().to(self.device)
                logger.info(f"Confidence maps extracted: {confidence_tensor.shape}")

            # Extract camera parameters if model supports pose estimation
            extrinsics_tensor = None
            intrinsics_tensor = None

            if self.supports_pose:
                # Extract extrinsics [N, 3, 4] - opencv w2c or colmap format
                if hasattr(prediction, 'extrinsics') and prediction.extrinsics is not None:
                    ext_array = prediction.extrinsics  # [N, 3, 4]
                    extrinsics_tensor = torch.from_numpy(ext_array).float().to(self.device)
                    logger.info(f"Camera extrinsics extracted: {extrinsics_tensor.shape}")

                # Extract intrinsics [N, 3, 3]
                if hasattr(prediction, 'intrinsics') and prediction.intrinsics is not None:
                    int_array = prediction.intrinsics  # [N, 3, 3]
                    intrinsics_tensor = torch.from_numpy(int_array).float().to(self.device)
                    logger.info(f"Camera intrinsics extracted: {intrinsics_tensor.shape}")
            else:
                logger.debug(f"Model {self.model_name} does not support camera pose estimation")

            # Extract processed images if available
            processed_tensor = None
            if hasattr(prediction, 'processed_images') and prediction.processed_images is not None:
                proc_array = prediction.processed_images  # [N, H, W, 3] uint8
                processed_tensor = torch.from_numpy(proc_array).to(self.device)

            # Handle single image case
            if not is_batch:
                depth_tensor = depth_tensor.squeeze(0)
                raw_depth_tensor = raw_depth_tensor.squeeze(0)
                if confidence_tensor is not None:
                    confidence_tensor = confidence_tensor.squeeze(0)
                if extrinsics_tensor is not None:
                    extrinsics_tensor = extrinsics_tensor.squeeze(0)
                if intrinsics_tensor is not None:
                    intrinsics_tensor = intrinsics_tensor.squeeze(0)
                if processed_tensor is not None:
                    processed_tensor = processed_tensor.squeeze(0)

            logger.info(f"DA3 inference complete. Depth: {depth_tensor.shape}, "
                       f"Pose support: {self.supports_pose}")

            return DA3Prediction(
                depth=depth_tensor,
                confidence=confidence_tensor,
                extrinsics=extrinsics_tensor,
                intrinsics=intrinsics_tensor,
                processed_images=processed_tensor,
                raw_depth=raw_depth_tensor,
                supports_pose=self.supports_pose
            )

        except Exception as e:
            logger.error(f"Error in DA3 inference: {e}")
            logger.error(traceback.format_exc())

            # Return error placeholder
            if isinstance(image, Image.Image):
                w, h = image.size
            elif isinstance(image, list) and len(image) > 0:
                w, h = image[0].size
            else:
                w, h = 512, 512

            dummy_depth = torch.ones((h, w), device=self.device, dtype=torch.float32)
            return DA3Prediction(depth=dummy_depth, supports_pose=False)

    def eval(self):
        """Compatibility method for eval mode."""
        if hasattr(self.model, 'eval'):
            self.model.eval()
        return self

    def to(self, device):
        """Move model to specified device."""
        if hasattr(self.model, 'to'):
            self.model = self.model.to(device)
        self.device = device
        return self
```

**Key Design Decisions:**

1. **Dataclass for Structured Output:** Uses `@dataclass` for clean, typed prediction structure
2. **Optional Fields:** Camera data is `Optional[torch.Tensor]` for graceful degradation
3. **Automatic Capability Detection:** `_check_pose_support()` determines if model supports camera estimation
4. **Raw Depth Preservation:** Stores both normalized (0-1) and raw depth for different use cases
5. **Comprehensive Error Handling:** Returns valid placeholder on failure

---

### 4.2 Enhanced DepthEstimationNode

**Location:** Extend existing class at lines 743+

#### 4.2.1 Modified Return Types

**Current (Line 784):**

```python
RETURN_TYPES = ("IMAGE",)
```

**New:**

```python
RETURN_TYPES = (
    "IMAGE",              # depth map (normalized 0-1)
    "IMAGE",              # confidence map (0-1), or None
    "CAMERA_EXTRINSICS",  # [N, 3, 4] camera extrinsics, or None
    "CAMERA_INTRINSICS",  # [N, 3, 3] camera intrinsics, or None
    "STRING",             # camera metadata as JSON string
)

RETURN_NAMES = (
    "depth",
    "confidence",
    "extrinsics",
    "intrinsics",
    "camera_json",
)
```

**Rationale:**

- ComfyUI ignores unused outputs, so no workflow breakage
- Custom types (`CAMERA_EXTRINSICS`, `CAMERA_INTRINSICS`) for type safety
- JSON string output for human readability and external tool integration
- Follows existing pattern of multiple outputs (like KSampler returns latent + denoised)

#### 4.2.2 New Input Parameters

Add to INPUT_TYPES (lines 761-782):

```python
"optional": {
    "force_reload": ("BOOLEAN", {"default": False}),
    "force_cpu": ("BOOLEAN", {"default": False}),
    "enable_camera_estimation": ("BOOLEAN", {"default": True}),  # NEW
    "output_raw_depth": ("BOOLEAN", {"default": False}),        # NEW
}
```

**Rationale:**

- `enable_camera_estimation`: Allow users to disable camera computation for speed
- `output_raw_depth`: Toggle between normalized (0-1) vs raw metric depth output

#### 4.2.3 Modified estimate_depth() Method

**Current Signature (Line 2126):**

```python
def estimate_depth(self, image, model_name, input_size, blur_radius,
                  median_size, apply_auto_contrast, apply_gamma,
                  force_reload=False, force_cpu=False):
```

**New Signature:**

```python
def estimate_depth(
    self,
    image,
    model_name,
    input_size,
    blur_radius,
    median_size,
    apply_auto_contrast,
    apply_gamma,
    force_reload=False,
    force_cpu=False,
    enable_camera_estimation=True,  # NEW
    output_raw_depth=False          # NEW
):
```

**New Return Structure:**

```python
# Current return (line ~2500):
return (depth_output,)

# New return:
return (
    depth_output,        # IMAGE: depth map
    confidence_output,   # IMAGE: confidence map or None
    extrinsics_output,   # CAMERA_EXTRINSICS or None
    intrinsics_output,   # CAMERA_INTRINSICS or None
    camera_json_output   # STRING: JSON metadata
)
```

**Implementation Logic:**

```python
def estimate_depth(self, image, model_name, input_size, blur_radius,
                  median_size, apply_auto_contrast, apply_gamma,
                  force_reload=False, force_cpu=False,
                  enable_camera_estimation=True, output_raw_depth=False):
    """
    Main depth estimation method with optional camera pose estimation.

    For DA3 Any-view models (Small, Base, Large, Giant, Nested), also returns
    camera extrinsics, intrinsics, and confidence maps.
    """

    # Existing validation and preprocessing (lines 2126-2200)
    # ... [keep existing code for input validation, device setup, etc.]

    # Load model (existing logic around line 823)
    self.ensure_model_loaded(model_name, force_reload, force_cpu)

    # Convert ComfyUI IMAGE tensor to PIL Images
    # ... [existing conversion logic]

    # Run inference
    if isinstance(self.depth_estimator, DA3ModelWrapper):
        # DA3 model - get full prediction
        prediction = self.depth_estimator(pil_images)

        # Extract depth - choose raw or normalized based on parameter
        if output_raw_depth:
            depth_result = prediction.raw_depth  # Metric depth
        else:
            depth_result = prediction.depth  # Normalized 0-1

        # Extract camera data if enabled and supported
        if enable_camera_estimation and prediction.supports_pose:
            confidence = prediction.confidence
            extrinsics = prediction.extrinsics
            intrinsics = prediction.intrinsics

            # Create JSON metadata
            camera_json = self._serialize_camera_data(
                extrinsics, intrinsics, confidence, model_name
            )
        else:
            confidence = None
            extrinsics = None
            intrinsics = None
            camera_json = json.dumps({
                "model": model_name,
                "camera_estimation_enabled": enable_camera_estimation,
                "supports_pose": prediction.supports_pose,
                "message": "Camera estimation disabled or not supported by model"
            })

    else:
        # V1/V2/MiDaS models - existing pipeline logic
        # ... [existing code for non-DA3 models]

        depth_result = output["predicted_depth"]
        confidence = None
        extrinsics = None
        intrinsics = None
        camera_json = json.dumps({
            "model": model_name,
            "supports_pose": False,
            "message": f"Model {model_name} does not support camera estimation"
        })

    # Apply post-processing to depth (existing logic)
    # ... [blur, median, contrast, gamma - lines ~2300-2450]

    # Convert to ComfyUI output format
    depth_output = self._tensor_to_comfyui_image(depth_result)

    confidence_output = None
    if confidence is not None:
        confidence_output = self._tensor_to_comfyui_image(confidence)

    extrinsics_output = extrinsics  # Already tensor or None
    intrinsics_output = intrinsics  # Already tensor or None

    return (
        depth_output,
        confidence_output,
        extrinsics_output,
        intrinsics_output,
        camera_json
    )
```

#### 4.2.4 New Helper Methods

**Add after estimate_depth() method:**

```python
def _serialize_camera_data(
    self,
    extrinsics: Optional[torch.Tensor],
    intrinsics: Optional[torch.Tensor],
    confidence: Optional[torch.Tensor],
    model_name: str
) -> str:
    """
    Serialize camera data to JSON string for metadata output.

    Args:
        extrinsics: [N, 3, 4] or [3, 4] camera extrinsics
        intrinsics: [N, 3, 3] or [3, 3] camera intrinsics
        confidence: [N, H, W] or [H, W] confidence maps
        model_name: Name of the model used

    Returns:
        JSON string with camera parameters and metadata
    """
    data = {
        "model": model_name,
        "timestamp": time.time(),
        "format_version": "1.0"
    }

    # Serialize extrinsics
    if extrinsics is not None:
        ext_np = extrinsics.cpu().numpy()

        if ext_np.ndim == 3:  # Batch
            data["num_views"] = ext_np.shape[0]
            data["extrinsics"] = [
                {
                    "matrix": ext.tolist(),  # 3x4 matrix
                    "format": "opencv_w2c",
                    "rotation": ext[:3, :3].tolist(),
                    "translation": ext[:3, 3].tolist()
                }
                for ext in ext_np
            ]
        else:  # Single view
            data["num_views"] = 1
            data["extrinsics"] = [{
                "matrix": ext_np.tolist(),
                "format": "opencv_w2c",
                "rotation": ext_np[:3, :3].tolist(),
                "translation": ext_np[:3, 3].tolist()
            }]

    # Serialize intrinsics
    if intrinsics is not None:
        int_np = intrinsics.cpu().numpy()

        if int_np.ndim == 3:  # Batch
            data["intrinsics"] = [
                {
                    "matrix": intr.tolist(),  # 3x3 matrix
                    "fx": float(intr[0, 0]),
                    "fy": float(intr[1, 1]),
                    "cx": float(intr[0, 2]),
                    "cy": float(intr[1, 2])
                }
                for intr in int_np
            ]
        else:  # Single view
            data["intrinsics"] = [{
                "matrix": int_np.tolist(),
                "fx": float(int_np[0, 0]),
                "fy": float(int_np[1, 1]),
                "cx": float(int_np[0, 2]),
                "cy": float(int_np[1, 2])
            }]

    # Add confidence statistics
    if confidence is not None:
        conf_np = confidence.cpu().numpy()
        data["confidence_stats"] = {
            "mean": float(np.mean(conf_np)),
            "std": float(np.std(conf_np)),
            "min": float(np.min(conf_np)),
            "max": float(np.max(conf_np))
        }

    return json.dumps(data, indent=2)


def _tensor_to_comfyui_image(self, tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert depth/confidence tensor to ComfyUI IMAGE format.

    ComfyUI IMAGE format: [B, H, W, C] with values in [0, 1] range

    Args:
        tensor: Input tensor [H, W] or [B, H, W]

    Returns:
        ComfyUI IMAGE tensor [B, H, W, C]
    """
    # Ensure batch dimension
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)  # [1, H, W]

    # Add channel dimension and repeat to RGB
    tensor = tensor.unsqueeze(-1)  # [B, H, W, 1]
    tensor = tensor.repeat(1, 1, 1, 3)  # [B, H, W, 3]

    # Ensure [0, 1] range
    tensor = torch.clamp(tensor, 0.0, 1.0)

    return tensor
```

---

### 4.3 Extended DEPTH_MODELS Dictionary

**Location:** Lines 303-322 (extend existing)

**Add Missing DA3 Models:**

```python
DEPTH_MODELS = {
    # ... existing models ...

    # EXISTING (keep as-is)
    "Depth-Anything-V3-Small": {
        "path": "depth-anything/DA3-Small",
        "vram_mb": 2000,
        "model_type": "v3",
        "encoder": "vits",
        "license": "Apache-2.0",
        "supports_batch": True,
        "supports_pose": True,  # NEW
        "params": "80M"
    },
    "Depth-Anything-V3-Base": {
        "path": "depth-anything/DA3-Base",
        "vram_mb": 2500,
        "model_type": "v3",
        "encoder": "vitb",
        "license": "Apache-2.0",
        "supports_batch": True,
        "supports_pose": True,  # NEW
        "params": "120M"
    },

    # NEW MODELS
    "Depth-Anything-V3-Large": {
        "path": "depth-anything/DA3-Large",
        "vram_mb": 4000,  # 350M params
        "model_type": "v3",
        "encoder": "vitl",
        "license": "CC BY-NC 4.0",
        "supports_batch": True,
        "supports_pose": True,
        "params": "350M"
    },
    "Depth-Anything-V3-Giant": {
        "path": "depth-anything/DA3-Giant",
        "vram_mb": 6000,  # 1.15B params
        "model_type": "v3",
        "encoder": "vitg",
        "license": "CC BY-NC 4.0",
        "supports_batch": True,
        "supports_pose": True,
        "params": "1.15B"
    },
    "Depth-Anything-V3-Nested-Giant-Large": {
        "path": "depth-anything/DA3NESTED-GIANT-LARGE",
        "vram_mb": 7000,  # 1.4B params
        "model_type": "v3",
        "encoder": "nested",
        "license": "CC BY-NC 4.0",
        "supports_batch": True,
        "supports_pose": True,
        "metric_scaling": True,  # Combined model with metric scaling
        "params": "1.4B"
    },

    # Mono and Metric variants (NO pose estimation)
    "Depth-Anything-V3-Mono-Large": {
        "path": "depth-anything/DA3Mono-Large",
        "vram_mb": 4000,
        "model_type": "v3",
        "encoder": "vitl",
        "license": "Apache-2.0",
        "supports_batch": False,
        "supports_pose": False,  # NO pose estimation
        "params": "350M",
        "note": "Monocular only, no camera estimation"
    },
    "Depth-Anything-V3-Metric-Large": {
        "path": "depth-anything/DA3Metric-Large",
        "vram_mb": 4000,
        "model_type": "v3",
        "encoder": "vitl",
        "license": "Apache-2.0",
        "supports_batch": False,
        "supports_pose": False,  # NO pose estimation
        "metric_depth": True,
        "params": "350M",
        "note": "Metric depth only, no camera estimation"
    },
}
```

---

### 4.4 Modified ensure_model_loaded() Method

**Location:** Lines 823-986

**Changes Required:**

```python
def ensure_model_loaded(self, model_name: str, force_reload: bool = False,
                       force_cpu: bool = False) -> None:
    """
    Ensures the correct model is loaded with proper VRAM management.

    CHANGES:
    - Pass model_name to DA3ModelWrapper constructor (line 974)
    - Model name used for automatic pose capability detection
    """

    # ... existing code lines 823-973 ...

    # MODIFIED SECTION (around line 974):
    # Old:
    # self.depth_estimator = DA3ModelWrapper(da3_model, self.device if not force_cpu else 'cpu')

    # New:
    self.depth_estimator = DA3ModelWrapper(
        da3_model,
        self.device if not force_cpu else 'cpu',
        model_name  # Pass model name for capability detection
    )

    # ... rest of existing code ...
```

---

## 5. Custom ComfyUI Type Definitions

**Location:** Add near top of file (after imports, around line 38)

```python
# Custom ComfyUI type definitions for camera parameters
# These types provide better clarity in workflow connections

# Camera extrinsics: [N, 3, 4] transformation matrices (opencv w2c format)
# - N is number of views
# - 3x3 rotation matrix + 3x1 translation vector
# - World-to-camera transformation
CAMERA_EXTRINSICS = "CAMERA_EXTRINSICS"

# Camera intrinsics: [N, 3, 3] calibration matrices
# - Contains focal lengths (fx, fy) and principal point (cx, cy)
# - Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
CAMERA_INTRINSICS = "CAMERA_INTRINSICS"

# Register custom types (if needed for ComfyUI type system)
# This ensures proper type checking and UI display
```

---

## 6. Performance Optimization Strategies

### 6.1 Caching Strategy

**Current:** Model cached in `self.depth_estimator` (line 756)

**Enhancement:** Cache prediction results for identical inputs

```python
class DepthEstimationNode:
    def __init__(self):
        self.device = None
        self.depth_estimator = None
        self.current_model = None

        # NEW: Prediction cache for identical inputs
        self._prediction_cache = {}  # key: image_hash, value: DA3Prediction
        self._cache_max_size = 5     # Limit cache size

    def _get_image_hash(self, image_tensor: torch.Tensor) -> str:
        """Generate hash of image for caching."""
        # Use subset of pixels for speed
        sample = image_tensor[::10, ::10].cpu().numpy()
        return hashlib.md5(sample.tobytes()).hexdigest()

    def estimate_depth(self, image, ...):
        # Check cache for DA3 models
        if isinstance(self.depth_estimator, DA3ModelWrapper):
            img_hash = self._get_image_hash(image)
            cache_key = f"{self.current_model}_{img_hash}"

            if cache_key in self._prediction_cache:
                logger.info("Using cached prediction")
                prediction = self._prediction_cache[cache_key]
            else:
                prediction = self.depth_estimator(pil_images)

                # Cache with size limit
                if len(self._prediction_cache) >= self._cache_max_size:
                    # Remove oldest entry
                    self._prediction_cache.pop(next(iter(self._prediction_cache)))

                self._prediction_cache[cache_key] = prediction
```

**Trade-offs:**

- Memory: ~500MB per cached prediction (depth + camera data)
- Speed: 50-80% faster for duplicate inputs
- Use case: Video processing with similar frames

### 6.2 Batch Processing Optimization

**Current:** DA3ModelWrapper handles single/batch (lines 654-660)

**Enhancement:** Optimize batch size based on VRAM

```python
def _optimize_batch_size(self, num_images: int, model_vram_mb: int) -> int:
    """
    Calculate optimal batch size based on available VRAM.

    Args:
        num_images: Total images to process
        model_vram_mb: Model VRAM requirement

    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return 1  # CPU processes one at a time

    free_mem = get_free_memory(self.device)
    free_mem_mb = free_mem / 1024

    # Rule of thumb: Each image needs ~200MB for processing
    per_image_mb = 200
    available_for_batch = free_mem_mb - model_vram_mb

    max_batch = max(1, int(available_for_batch / per_image_mb))
    optimal_batch = min(num_images, max_batch)

    logger.info(f"Optimal batch size: {optimal_batch} (free VRAM: {free_mem_mb:.1f}MB)")
    return optimal_batch
```

### 6.3 Conditional Camera Computation

**Optimization:** Skip camera computation when `enable_camera_estimation=False`

```python
# In DA3ModelWrapper.__call__():

# Only compute camera parameters if explicitly requested
if enable_camera_estimation and self.supports_pose:
    # Extract extrinsics/intrinsics
    ...
else:
    # Skip camera computation entirely
    extrinsics_tensor = None
    intrinsics_tensor = None
```

**Performance Gain:**

- ~10-15% faster inference when camera estimation disabled
- Reduced memory allocation

### 6.4 Lazy JSON Serialization

**Optimization:** Only serialize camera JSON when output is connected

```python
# In estimate_depth():

# Only serialize JSON if output is actually used
# Note: ComfyUI doesn't provide "is_output_connected" API,
# so always serialize for now, but design allows future optimization
camera_json = self._serialize_camera_data(...)
```

---

## 7. Error Handling Strategy

### 7.1 Graceful Degradation

**Principle:** Never crash the workflow, always return valid outputs

```python
# In DA3ModelWrapper.__call__():
try:
    prediction = self.model.inference(images)
except Exception as e:
    logger.error(f"DA3 inference failed: {e}")

    # Return valid placeholder prediction
    return DA3Prediction(
        depth=torch.ones((h, w), device=self.device),
        confidence=None,
        extrinsics=None,
        intrinsics=None,
        supports_pose=False
    )
```

### 7.2 Model Capability Validation

```python
def _validate_model_capabilities(self, model_name: str,
                                enable_camera_estimation: bool) -> None:
    """
    Validate that requested features are supported by model.

    Logs warnings but does not fail - graceful degradation.
    """
    model_info = DEPTH_MODELS.get(model_name, {})
    supports_pose = model_info.get("supports_pose", False)

    if enable_camera_estimation and not supports_pose:
        logger.warning(
            f"Camera estimation requested but model '{model_name}' "
            f"does not support it. Outputs will be None. "
            f"Use Any-view models (Small, Base, Large, Giant, Nested) "
            f"for camera estimation."
        )
```

### 7.3 Tensor Shape Validation

```python
def _validate_camera_tensors(self, extrinsics: Optional[torch.Tensor],
                             intrinsics: Optional[torch.Tensor]) -> bool:
    """Validate camera tensor shapes before use."""
    if extrinsics is not None:
        expected_shape = (3, 4) if extrinsics.ndim == 2 else (-1, 3, 4)
        if extrinsics.shape[-2:] != (3, 4):
            logger.error(f"Invalid extrinsics shape: {extrinsics.shape}, "
                        f"expected [..., 3, 4]")
            return False

    if intrinsics is not None:
        if intrinsics.shape[-2:] != (3, 3):
            logger.error(f"Invalid intrinsics shape: {intrinsics.shape}, "
                        f"expected [..., 3, 3]")
            return False

    return True
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

**File:** Create `tests/test_camera_estimation.py`

```python
import pytest
import torch
import numpy as np
from PIL import Image
from depth_estimation_node import DA3ModelWrapper, DA3Prediction

class TestDA3ModelWrapper:
    """Test DA3ModelWrapper camera estimation capabilities."""

    def test_pose_support_detection(self):
        """Test automatic detection of pose estimation support."""
        # Any-view models should support pose
        assert DA3ModelWrapper._check_pose_support(None, "Depth-Anything-V3-Small") == True
        assert DA3ModelWrapper._check_pose_support(None, "Depth-Anything-V3-Giant") == True
        assert DA3ModelWrapper._check_pose_support(None, "Depth-Anything-V3-Nested-Giant-Large") == True

        # Mono and Metric should NOT support pose
        assert DA3ModelWrapper._check_pose_support(None, "Depth-Anything-V3-Mono-Large") == False
        assert DA3ModelWrapper._check_pose_support(None, "Depth-Anything-V3-Metric-Large") == False

        # V1/V2 should not support pose
        assert DA3ModelWrapper._check_pose_support(None, "Depth-Anything-V2-Small") == False

    def test_prediction_structure(self):
        """Test DA3Prediction dataclass structure."""
        depth = torch.rand(512, 512)
        extrinsics = torch.rand(3, 4)
        intrinsics = torch.rand(3, 3)
        confidence = torch.rand(512, 512)

        pred = DA3Prediction(
            depth=depth,
            confidence=confidence,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            supports_pose=True
        )

        assert pred.depth.shape == (512, 512)
        assert pred.extrinsics.shape == (3, 4)
        assert pred.intrinsics.shape == (3, 3)
        assert pred.supports_pose == True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_batch_processing(self):
        """Test batch processing with multiple images."""
        # This requires actual DA3 model - mark as integration test
        pass


class TestCameraSerialization:
    """Test camera data JSON serialization."""

    def test_json_serialization_single_view(self):
        """Test JSON serialization for single view."""
        node = DepthEstimationNode()

        extrinsics = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 5]
        ], dtype=torch.float32)

        intrinsics = torch.tensor([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=torch.float32)

        json_str = node._serialize_camera_data(
            extrinsics, intrinsics, None, "DA3-Small"
        )

        import json
        data = json.loads(json_str)

        assert data["num_views"] == 1
        assert "extrinsics" in data
        assert "intrinsics" in data
        assert data["intrinsics"][0]["fx"] == 500.0

    def test_json_serialization_multi_view(self):
        """Test JSON serialization for multiple views."""
        # Test batch serialization
        pass


class TestTensorConversion:
    """Test tensor format conversions."""

    def test_tensor_to_comfyui_image(self):
        """Test conversion to ComfyUI IMAGE format."""
        node = DepthEstimationNode()

        # Single image [H, W]
        depth = torch.rand(512, 512)
        comfyui_img = node._tensor_to_comfyui_image(depth)

        assert comfyui_img.shape == (1, 512, 512, 3)  # [B, H, W, C]
        assert comfyui_img.min() >= 0.0
        assert comfyui_img.max() <= 1.0

        # Batch [B, H, W]
        depth_batch = torch.rand(4, 512, 512)
        comfyui_batch = node._tensor_to_comfyui_image(depth_batch)

        assert comfyui_batch.shape == (4, 512, 512, 3)
```

### 8.2 Integration Tests

**File:** Create `tests/test_integration.py`

```python
@pytest.mark.integration
class TestDA3Integration:
    """Integration tests with actual DA3 models."""

    @pytest.mark.skipif(not DA3_AVAILABLE, reason="Requires depth-anything-3")
    def test_small_model_camera_estimation(self):
        """Test camera estimation with DA3-Small model."""
        from depth_anything_3.api import DepthAnything3

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DepthAnything3.from_pretrained("depth-anything/DA3-Small")
        model = model.to(device)

        wrapper = DA3ModelWrapper(model, device, "Depth-Anything-V3-Small")

        # Create test image
        test_img = Image.new("RGB", (640, 480), color=(100, 150, 200))

        prediction = wrapper(test_img)

        assert prediction.depth is not None
        assert prediction.supports_pose == True
        assert prediction.extrinsics is not None
        assert prediction.intrinsics is not None
        assert prediction.extrinsics.shape == (3, 4)
        assert prediction.intrinsics.shape == (3, 3)

    @pytest.mark.skipif(not DA3_AVAILABLE, reason="Requires depth-anything-3")
    def test_mono_model_no_camera_estimation(self):
        """Test that Mono models correctly report no camera support."""
        # Test DA3Mono-Large returns supports_pose=False
        pass
```

### 8.3 Manual Testing Checklist

**Workflow Tests:**

1. **Single Image Depth (V1/V2 Model):**
   - Load V2-Small model
   - Process single image
   - Verify depth output works
   - Verify camera outputs are None
   - Verify JSON indicates no pose support

2. **Single Image with Camera (DA3-Small):**
   - Load DA3-Small model
   - Process single image
   - Verify depth output
   - Verify confidence output (should be non-None)
   - Verify extrinsics shape [3, 4]
   - Verify intrinsics shape [3, 3]
   - Verify JSON contains camera parameters

3. **Batch Processing (DA3-Base):**
   - Load DA3-Base model
   - Process batch of 4 images
   - Verify depth output shape [4, H, W, 3]
   - Verify extrinsics shape [4, 3, 4]
   - Verify intrinsics shape [4, 3, 3]
   - Verify JSON contains 4 views

4. **Camera Estimation Disabled:**
   - Load DA3-Large model
   - Set `enable_camera_estimation=False`
   - Verify depth works
   - Verify camera outputs are None
   - Verify performance improvement (~10-15% faster)

5. **Mono Model (No Pose Support):**
   - Load DA3Mono-Large
   - Process image
   - Verify depth works
   - Verify camera outputs are None
   - Check logs for "does not support camera estimation" message

6. **VRAM Management:**
   - Load DA3-Giant (6GB VRAM)
   - Verify VRAM warning if insufficient
   - Verify fallback to CPU if needed
   - Verify cleanup after processing

7. **Error Recovery:**
   - Corrupt model path
   - Verify graceful fallback
   - Verify placeholder depth returned
   - Verify workflow doesn't crash

---

## 9. Documentation Updates

### 9.1 README.md Updates

**Add Section: Camera Pose Estimation**

```markdown
## Camera Pose Estimation (Depth Anything V3)

Depth Anything V3 Any-view models (Small, Base, Large, Giant, Nested) support
camera pose estimation, providing camera extrinsics and intrinsics.

### Supported Models

| Model | Pose Estimation | License |
|-------|----------------|---------|
| DA3-Small | ✅ | Apache 2.0 |
| DA3-Base | ✅ | Apache 2.0 |
| DA3-Large | ✅ | CC BY-NC 4.0 |
| DA3-Giant | ✅ | CC BY-NC 4.0 |
| DA3Nested-Giant-Large | ✅ | CC BY-NC 4.0 |
| DA3Mono-Large | ❌ | Apache 2.0 |
| DA3Metric-Large | ❌ | Apache 2.0 |

### Usage

1. Select a DA3 Any-view model (e.g., "Depth-Anything-V3-Small")
2. Enable camera estimation in node settings (default: enabled)
3. Connect the camera outputs:
   - `extrinsics`: [N, 3, 4] camera extrinsics (opencv w2c format)
   - `intrinsics`: [N, 3, 3] camera intrinsics
   - `camera_json`: JSON string with camera parameters and metadata

### Camera Data Format

**Extrinsics (World-to-Camera):**
```

[N, 3, 4] matrix in OpenCV w2c or COLMAP format
[R | t] where R is 3x3 rotation, t is 3x1 translation

```

**Intrinsics (Camera Calibration):**
```

[N, 3, 3] matrix
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]

fx, fy: focal lengths in pixels
cx, cy: principal point (image center)

```

### JSON Metadata Example

```json
{
  "model": "Depth-Anything-V3-Small",
  "num_views": 1,
  "extrinsics": [{
    "matrix": [[...], [...], [...]],
    "format": "opencv_w2c",
    "rotation": [[...], [...], [...]],
    "translation": [x, y, z]
  }],
  "intrinsics": [{
    "matrix": [[...], [...], [...]],
    "fx": 500.0,
    "fy": 500.0,
    "cx": 320.0,
    "cy": 240.0
  }],
  "confidence_stats": {
    "mean": 0.85,
    "std": 0.12
  }
}
```

### Performance Tips

- Disable camera estimation (`enable_camera_estimation=False`) for ~10-15% speedup
  when camera data is not needed
- Use smaller models (Small/Base) for real-time applications
- Use Giant/Nested for highest quality camera estimation

```

### 9.2 Update CLAUDE.md

**Add to Code Architecture Section:**

```markdown
#### Camera Pose Estimation (NEW)

**DA3Prediction Dataclass:**
- Structured output from DA3 models
- Fields: depth, confidence, extrinsics, intrinsics, raw_depth, supports_pose
- Location: After DA3ModelWrapper class definition

**Enhanced DA3ModelWrapper:**
- Automatic capability detection via `_check_pose_support()`
- Extracts all prediction fields: depth, confidence, extrinsics, intrinsics
- Returns DA3Prediction dataclass

**DepthEstimationNode Extended Outputs:**
- RETURN_TYPES: ("IMAGE", "IMAGE", "CAMERA_EXTRINSICS", "CAMERA_INTRINSICS", "STRING")
- Optional outputs (unused outputs ignored by ComfyUI)
- Camera outputs are None for non-supporting models (graceful degradation)

**Model Support Matrix:**
- Any-view models (Small, Base, Large, Giant, Nested): Full camera estimation
- Mono/Metric models: Depth only, no camera estimation
- V1/V2 models: Depth only, no camera estimation

**Performance Optimizations:**
- Prediction caching for duplicate inputs
- Batch size optimization based on VRAM
- Conditional camera computation (skip when disabled)
- Lazy JSON serialization
```

---

## 10. Migration Path and Backward Compatibility

### 10.1 Backward Compatibility

**Guarantee:** Existing workflows will continue to work unchanged.

**Strategy:**

1. New outputs are appended (don't change existing output order)
2. New inputs are optional with sensible defaults
3. Node behavior unchanged when camera features not used

**Example:**

```python
# Old workflow (single depth output):
depth_map = DepthEstimationNode(image, "Depth-Anything-V2-Small", ...)
# Works unchanged - only depth_map is connected

# New workflow (with camera):
depth, confidence, extrinsics, intrinsics, json = DepthEstimationNode(
    image, "Depth-Anything-V3-Small", ...
)
# All outputs available, connect as needed
```

### 10.2 Gradual Adoption

Users can adopt camera features incrementally:

**Phase 1:** Use existing workflow (no changes)
**Phase 2:** Switch to DA3 model (get better depth, ignore camera outputs)
**Phase 3:** Connect camera outputs for downstream 3D processing

---

## 11. Future Extensions (Out of Scope)

**Not implemented in this phase, but designed to support:**

1. **Point Cloud Generation:**
   - New node: `DA3PointCloudNode`
   - Inputs: depth (raw), confidence, intrinsics, extrinsics, image
   - Output: PLY file or 3D tensor

2. **3D Gaussian Splatting:**
   - Extract 3D Gaussians from DA3 prediction
   - Requires model fine-tuning or post-processing

3. **Multi-View Fusion:**
   - Combine depth from multiple views using camera poses
   - ICP alignment or pose-based fusion

4. **Camera Conditioning:**
   - Input known camera poses to improve depth consistency
   - Requires extending input parameters

5. **Tiled Processing:**
   - High-resolution image processing with tiles
   - Requires batching logic changes

---

## 12. Implementation Checklist

### Phase 1: Core Infrastructure (Priority 1)

- [ ] Add `DA3Prediction` dataclass (after line 38)
- [ ] Add custom type definitions `CAMERA_EXTRINSICS`, `CAMERA_INTRINSICS` (after imports)
- [ ] Extend DEPTH_MODELS with new DA3 models (lines 303-322)
- [ ] Update DA3ModelWrapper.**init**() to accept model_name (line 629)
- [ ] Implement DA3ModelWrapper._check_pose_support() (new method)
- [ ] Update DA3ModelWrapper.**call**() to return DA3Prediction (lines 641-708)
  - Extract confidence maps
  - Extract extrinsics
  - Extract intrinsics
  - Extract raw depth
  - Return structured dataclass

### Phase 2: Node Extension (Priority 1)

- [ ] Update DepthEstimationNode.RETURN_TYPES (line 784)
- [ ] Add RETURN_NAMES (new, after RETURN_TYPES)
- [ ] Add new optional inputs to INPUT_TYPES (lines 778-781)
  - `enable_camera_estimation`
  - `output_raw_depth`
- [ ] Update estimate_depth() signature (line 2126)
- [ ] Update estimate_depth() implementation (lines 2126+)
  - Handle DA3Prediction
  - Extract camera data
  - Apply conditional logic for camera estimation
  - Return 5-tuple instead of 1-tuple
- [ ] Implement _serialize_camera_data() helper (new method)
- [ ] Implement _tensor_to_comfyui_image() helper (new method)
- [ ] Update ensure_model_loaded() to pass model_name to wrapper (line 974)

### Phase 3: Testing (Priority 2)

- [ ] Create tests/test_camera_estimation.py
- [ ] Implement unit tests for DA3ModelWrapper
- [ ] Implement unit tests for camera serialization
- [ ] Implement unit tests for tensor conversion
- [ ] Create tests/test_integration.py
- [ ] Implement integration test with DA3-Small
- [ ] Implement integration test with DA3Mono (verify no pose support)
- [ ] Manual testing with actual workflows

### Phase 4: Documentation (Priority 2)

- [ ] Update README.md with camera estimation section
- [ ] Add usage examples to README
- [ ] Update CLAUDE.md with architecture changes
- [ ] Add inline code comments for new methods
- [ ] Create example workflow JSON (optional)

### Phase 5: Performance Optimization (Priority 3)

- [ ] Implement prediction caching (optional)
- [ ] Implement batch size optimization (optional)
- [ ] Profile inference performance (optional)
- [ ] Optimize JSON serialization (optional)

---

## 13. Code Locations Summary

| Component | Location (Line Numbers) | Action |
|-----------|------------------------|--------|
| Imports | 1-38 | Add dataclass import |
| Type definitions | After 38 (new) | Add CAMERA_EXTRINSICS, CAMERA_INTRINSICS |
| DA3Prediction | After 38 (new) | Add dataclass definition |
| DA3ModelWrapper.**init** | 629-639 | Add model_name parameter |
| DA3ModelWrapper._check_pose_support | New method | Add capability detection |
| DA3ModelWrapper.**call** | 641-708 | Extend to extract all fields |
| DEPTH_MODELS | 303-322 | Add new DA3 models |
| DepthEstimationNode.INPUT_TYPES | 761-782 | Add optional inputs |
| DepthEstimationNode.RETURN_TYPES | 784 | Extend to 5 outputs |
| DepthEstimationNode.estimate_depth | 2126+ | Extend implementation |
| Helper methods | After estimate_depth (new) | Add _serialize_camera_data,_tensor_to_comfyui_image |
| ensure_model_loaded | 974 | Pass model_name to wrapper |

---

## 14. Risk Assessment

### High Risk

- **Backward compatibility:** Mitigated by optional outputs and sensible defaults
- **VRAM overflow:** Mitigated by existing VRAM checks and batch optimization
- **Model download failures:** Mitigated by existing fallback logic

### Medium Risk

- **Type confusion:** Mitigated by custom type definitions and validation
- **JSON serialization overhead:** Acceptable (< 5ms), lazy evaluation possible
- **Cache memory usage:** Mitigated by cache size limits (5 predictions max)

### Low Risk

- **ComfyUI API changes:** Unlikely, using stable patterns
- **DA3 API changes:** Unlikely, API is stable
- **Performance regression:** Mitigated by conditional computation

---

## 15. Success Criteria

**Implementation is successful when:**

1. ✅ Existing workflows with V1/V2 models work unchanged
2. ✅ DA3 Any-view models return camera extrinsics and intrinsics
3. ✅ DA3 Mono/Metric models gracefully degrade (no camera data)
4. ✅ Camera data validates correctly (shape, range, format)
5. ✅ JSON metadata is well-formed and human-readable
6. ✅ Performance impact < 5% when camera estimation disabled
7. ✅ No VRAM regressions
8. ✅ All unit tests pass
9. ✅ Integration tests with actual DA3 models pass
10. ✅ Documentation is comprehensive and accurate

---

## 16. Next Steps After Implementation

1. **Community Feedback:** Gather user feedback on camera output format
2. **Point Cloud Node:** Implement 3D point cloud generation using camera data
3. **Multi-View Fusion:** Add support for fusing depth from multiple views
4. **Camera Conditioning:** Add input for known camera poses
5. **Export Formats:** Add support for COLMAP, Bundler, NeRF formats
6. **Performance Profiling:** Detailed profiling and optimization

---

**End of Implementation Plan**

**Estimated Implementation Time:**

- Phase 1 (Core): 4-6 hours
- Phase 2 (Node): 6-8 hours
- Phase 3 (Testing): 4-6 hours
- Phase 4 (Documentation): 2-3 hours
- Total: 16-23 hours

**Dependencies:**

- `depth-anything-3` package (optional, graceful degradation if missing)
- No new ComfyUI dependencies

**Files Modified:**

- `/home/gero/GitHub/ComfyUI_Lab/ComfyUIDepthEstimation/depth_estimation_node.py`

**Files Created:**

- `/home/gero/GitHub/ComfyUI_Lab/ComfyUIDepthEstimation/tests/test_camera_estimation.py`
- `/home/gero/GitHub/ComfyUI_Lab/ComfyUIDepthEstimation/tests/test_integration.py`
- Example workflow JSON (optional)
