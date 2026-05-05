---
title: "Chapter 1 — Image Fundamentals"
---

[← Back to Table of Contents](./README.md)

# Chapter 1 — Image Fundamentals

> *"An image is a 3D tensor. Everything else is interpretation."*

## Images as Tensors

> **Planned content:** How a digital image is stored in memory. The `[H, W, C]` convention (HWC) in NumPy/PIL vs. `[C, H, W]` (CHW) in PyTorch vs. `[B, C, H, W]` (NCHW) in batched training. uint8 vs. float32 storage. Memory layout (row-major, channel-major). Why the ordering matters for GPU memory access patterns.

> **📊 Planned diagram:** Visual decomposition of a `[3, 224, 224]` tensor into R, G, B channel planes with pixel value ranges.

> **📊 Planned diagram:** Memory layout illustration — NCHW vs. NHWC, and why deep learning frameworks default to NCHW on CUDA.

```python
import torch
from PIL import Image
import numpy as np

img_pil = Image.open("cat.jpg")          # PIL: HWC, uint8
img_np  = np.array(img_pil)              # [H, W, 3], dtype=uint8, range [0, 255]
img_t   = torch.from_numpy(img_np)       # [H, W, 3], dtype=uint8
img_t   = img_t.permute(2, 0, 1)        # [3, H, W], dtype=uint8  — CHW
img_t   = img_t.float() / 255.0         # [3, H, W], dtype=float32, range [0, 1]
img_t   = img_t.unsqueeze(0)            # [1, 3, H, W]  — add batch dimension
```

## Color Spaces

> **Planned content:** RGB (additive, camera-native). BGR (OpenCV convention — a historical quirk). HSV/HSL (hue-saturation-value, perceptually intuitive, used in augmentation). YCbCr (luminance-chrominance, used in JPEG compression and video). LAB (perceptually uniform, used in style transfer and color correction). Grayscale conversion. When to convert and why.

> **📊 Planned diagram:** Color space comparison — same image rendered in RGB, HSV, LAB, YCbCr channels. Color wheel for HSV.

> **📊 Planned diagram:** Gamut visualization — LAB color space as a 3D volume.

```python
import cv2

img_bgr  = cv2.imread("cat.jpg")                     # [H, W, 3], BGR uint8
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # [H, W, 3], RGB
img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)  # [H, W, 3], HSV
img_lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)  # [H, W, 3], LAB
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) # [H, W],    grayscale
```

## Preprocessing Pipelines

> **Planned content:** The canonical preprocessing pipeline for training: load → decode → resize → random crop → color jitter → normalize. For inference: load → decode → resize → center crop → normalize. Why the order matters (resize before crop vs. crop before resize). Batching and collation.

> **📊 Planned diagram (flowchart):** Training preprocessing pipeline vs. inference preprocessing pipeline with tensor shapes at each step.

### Resizing and Padding Strategies

> **Planned content:** Simple resize (distorts aspect ratio). Letterboxing/padding (preserves aspect ratio, used by YOLO). Multi-scale training. Image pyramids. Interpolation methods (nearest, bilinear, bicubic, lanczos) and when to use each.

> **📊 Planned diagram:** Visual comparison of resize strategies — stretch vs. letterbox vs. center-crop, showing how object proportions are affected.

### Normalization

> **Planned content:** Why normalize? Zero-centering and unit variance for stable gradients. ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` — where these numbers come from. Per-channel vs. global normalization. Instance normalization at inference time.

```python
import torchvision.transforms as T

# Standard ImageNet normalization
normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406],  # ImageNet channel means
    std =[0.229, 0.224, 0.225]   # ImageNet channel stds
)

train_transform = T.Compose([
    T.RandomResizedCrop(224),    # → [3, 224, 224]
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.ToTensor(),                # PIL [H, W, 3] → Tensor [3, H, W], /255
    normalize,                   # → [3, 224, 224], normalized
])
```

## Data Types and Precision

> **Planned content:** uint8 (0–255, storage), float32 (training, full precision), float16/bfloat16 (mixed precision training and inference), int8 (quantized inference). Memory footprint: a 3×224×224 image is 150K float32 bytes or 37.5K uint8 bytes. Precision implications at each stage.

> **📊 Planned table:** Data type comparison — range, precision, memory per element, when used.

## Image Quality and Artifacts

> **Planned content:** JPEG compression artifacts (blocking, ringing). PNG lossless. Noise models (Gaussian, Poisson/shot noise, salt-and-pepper). Blur (motion, defocus). Lens distortion. Why understanding degradations matters for robust models and augmentation design.

## Camera Models and Intrinsics

> **Planned content:** Pinhole camera model. Focal length, principal point, skew. The intrinsic matrix K. Radial and tangential distortion. Undistortion. Homogeneous coordinates. Why this matters for depth estimation, 3D detection, and NeRF.

> **📊 Planned diagram:** Pinhole camera model — ray from 3D point through lens to image plane, with formula labels.

$$\begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = \frac{1}{Z} \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} X \\ Y \\ Z \end{pmatrix}$$

## Frequency Domain Fundamentals

> **Planned content:** Discrete Fourier Transform (DFT) of images. Spatial frequency and what it means visually. Low-frequency = global structure, high-frequency = edges/texture. Fourier features in position encoding (NeRF, FNO). Why CNNs have a bias toward low-frequency functions. Spectral leakage and windowing.

> **📊 Planned diagram:** Image in spatial domain vs. its Fourier magnitude spectrum — showing how edges create high-frequency components.

## Geometric Transformations

> **Planned content:** Affine transformations (rotation, scale, shear, translation) as matrix multiplications. Perspective transforms (homographies). Thin-plate spline (TPS) transforms. Grid sampling (`torch.nn.functional.grid_sample`). Why these matter for data augmentation and spatial transformer networks.

> **📊 Planned diagram:** Affine transformation matrix decomposition — rotation, scale, shear components visualized.

**Next: [Chapter 2 — Image Classification →](./02_image_classification.md)**

---
*Last updated: May 2026*
