---
title: "Chapter 33 — Data Augmentation"
---

[← Back to Table of Contents](./README.md)

# Chapter 33 — Data Augmentation

> *"Augmentation is domain knowledge encoded as code — every transform is a hypothesis about invariances in your task."*

## Classical Augmentation

> **Planned content:** Random horizontal flip (classification, detection — but NOT text recognition). Random crop + resize (random resized crop). Color jitter: brightness, contrast, saturation, hue. Gaussian blur. Grayscale conversion. Random rotation (±30°). Random erasing / cutout. Composing augmentations with `torchvision.transforms`.

> **📊 Planned diagram:** Augmentation gallery — 8 variants of the same image showing different augmentation applied.

```python
import torchvision.transforms as T

train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0)),   # [3, 224, 224]
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    T.ToTensor(),                                   # [3, 224, 224] float [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.25)                         # [3, 224, 224]
])
```

## MixUp: Linear Interpolation (2017)

> **Planned content:** Mix two training examples: `x_mixed = λ*x_a + (1-λ)*x_b`, `y_mixed = λ*y_a + (1-λ)*y_b`. λ ~ Beta(α, α). Soft labels improve calibration. Why mixing in input space trains more stable models. Applied in classification, detection (MixUp on instance level).

> **📊 Planned diagram:** MixUp — two images blended with λ=0.4, resulting mixed image, and mixed one-hot target showing partial labels for both classes.

```python
def mixup_data(x, y, alpha=1.0):
    """
    x: [B, C, H, W] — batch of images
    y: [B, C]       — one-hot labels
    Returns: mixed_x [B, C, H, W], mixed_y [B, C], lam
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    rand_index = torch.randperm(x.size(0))      # [B] — random permutation
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]   # [B, C, H, W]
    mixed_y = lam * y + (1 - lam) * y[rand_index, :]   # [B, C]
    return mixed_x, mixed_y, lam
```

## CutMix: Paste Patches (2019)

> **Planned content:** Cut a random rectangular region from one image, paste into another. Labels mixed proportionally to area. Stronger spatial regularization than MixUp. Encourages recognizing objects from partial views.

> **📊 Planned diagram:** CutMix — image A with rectangular region cut + image B's content pasted → mixed image. Label = area fraction * class_A + (1-area fraction) * class_B.

## Mosaic: YOLO's Secret Weapon (2020)

> **Planned content:** Tile 4 images into a 2×2 mosaic. Each image scaled/cropped to a quadrant. Labels adjusted for new positions. Effect: 4× more objects per image, varied scales and contexts. Critical for YOLO training on small objects. Introduced in YOLOv4.

> **📊 Planned diagram:** Mosaic augmentation — 4 input images → 2×2 tiled output with adjusted bounding boxes.

## AutoAugment and RandAugment

> **Planned content:** AutoAugment: use RL to search for the best augmentation policy on validation set. Discrete set of operations (shear, translate, rotate, flip, brightness, etc.) with magnitude and probability. RandAugment: simplified — randomly apply N random operations from a fixed set with a single magnitude M. TrivialAugment: even simpler — sample one random operation per image.

> **📊 Planned diagram:** RandAugment pipeline — image → randomly select N ops from pool → apply sequentially → augmented image.

## Copy-Paste for Segmentation (2021)

> **Planned content:** Copy instances (objects + masks) from one image, paste onto another. Critical for instance segmentation, especially rare objects. Scale and flip the pasted instance. Helps with out-of-distribution instances in new backgrounds.

> **📊 Planned diagram:** Copy-paste augmentation — source image instances extracted with masks → pasted onto target image → updated instance mask annotations.

## Test-Time Augmentation (TTA)

> **Planned content:** At inference: apply multiple augmentations, run model on each, average predictions. Common TTA: flip (horizontal), multi-scale, rotation at 0°/90°/270°. Accuracy improvement: +0.5-2% typically. Cost: K× more inference. When it's worth it.

## Detection-Specific Augmentation

> **Planned content:** Random horizontal flip (adjust box coordinates). Random scale (resize with new stride). Multi-scale training. Mixup at instance level (YOLO). Albumentations library for coordinated image + bbox augmentation.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.RandomResizedCrop(640, 640),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

**Next: [Chapter 34 — Training Recipes →](./34_training_recipes.md)**

---
*Last updated: May 2026*
