---
title: "Chapter 18 — Classification Architectures"
---

[← Back to Table of Contents](./README.md)

# Chapter 18 — Classification Architectures

> *"Every architecture is a hypothesis about what matters in images — depth, skip connections, scale, or attention."*

## AlexNet and the Deep Learning Revolution (2012)

> **Planned content:** 5 conv layers + 3 FC layers, ReLU, dropout, data augmentation, GPU training. Why it was revolutionary: 10.8% top-5 error vs. 25.8% for runner-up on ImageNet 2012. Local response normalization. Landmark paper anatomy.

> **📊 Planned diagram:** AlexNet architecture with tensor shapes — `[B, 3, 227, 227]` → 5 conv blocks → `[B, 256, 6, 6]` → flatten → 3 FC → `[B, 1000]`.

## VGG: Depth and Simplicity (2014)

> **Planned content:** VGG-16, VGG-19. Only 3×3 convolutions. Very deep (16-19 layers). 138M parameters — large by modern standards. Why uniform architecture is good for transfer learning. Still used as backbone in feature extraction.

## ResNet: Residual Connections (2015)

> **Planned content:** The degradation problem: deeper networks perform worse without skip connections (not overfitting — training accuracy also degrades). Solution: `y = F(x) + x`. Basic block (2 conv) vs. Bottleneck block (1×1 → 3×3 → 1×1 for deep ResNets). ResNet-18/34/50/101/152. Pre-activation ResNet (ResNetV2). Wide ResNet. ResNeXt: grouped convolutions.

> **📊 Planned diagram:** ResNet basic block and bottleneck block — tensor shapes `[B, 64, 56, 56]` → block → `[B, 64, 56, 56]` with shortcut connection.

> **📊 Planned diagram:** ResNet architecture pyramid — stage 1 (64ch, stride 4) → stage 2 (128ch, stride 2) → stage 3 (256ch) → stage 4 (512ch) → GAP → FC.

## MobileNet: Efficiency (v1/v2/v3)

> **Planned content:** MobileNetV1: depthwise separable convolutions throughout. MobileNetV2: inverted residuals + linear bottlenecks. Why linear activation in bottleneck prevents information loss. ReLU6. MobileNetV3: NAS-designed, hard-swish, squeeze-and-excite.

> **📊 Planned diagram:** Inverted residual block — narrow → wide → narrow (reversed from ResNet), with linear bottleneck and SE block.

## EfficientNet: Compound Scaling (2019)

> **Planned content:** Compound scaling: simultaneously scale depth, width, and resolution using a fixed coefficient φ. NAS to find the baseline architecture. EfficientNet-B0 to B7. EfficientNetV2: progressive training, smaller early-stage conv. Pareto frontier vs. ResNet.

> **📊 Planned diagram:** Compound scaling illustration — depth α^φ, width β^φ, resolution γ^φ scaling from B0.

> **📊 Planned diagram:** EfficientNet accuracy vs. FLOPs Pareto curve — comparing with ResNet, MobileNet, NASNet.

## Vision Transformer (ViT): Patches as Tokens (2020)

> **Planned content:** Divide image into 16×16 patches → flatten → linear projection to `d_model`. Add positional embeddings. Prepend CLS token. Pass through standard Transformer encoder (L layers). CLS token → classification head. Requires large data (ViT-L trained on JFT-300M). DeiT: train ViT on ImageNet with distillation.

> **📊 Planned diagram:** ViT architecture — `[B, 3, 224, 224]` → patch embedding `[B, 196, 768]` → add CLS token `[B, 197, 768]` → L × Transformer block → CLS token → head `[B, 1000]`.

```python
# ViT patch embedding
B, C, H, W = 8, 3, 224, 224
patch_size = 16
n_patches = (H // patch_size) * (W // patch_size)  # 196

patch_embed = nn.Conv2d(C, 768, kernel_size=patch_size, stride=patch_size)
# Equivalent to flattening 16×16 patches and projecting

x = torch.randn(B, C, H, W)         # [8, 3, 224, 224]
x = patch_embed(x)                   # [8, 768, 14, 14]
x = x.flatten(2).transpose(1, 2)    # [8, 196, 768]
```

## Swin Transformer: Hierarchical Windows (2021)

> **Planned content:** Key insight: ViT global attention is O(n²) — impractical for dense prediction. Swin: local window attention (7×7 patches per window) — O(n). Shifted windows between layers for cross-window communication. Hierarchical feature maps (like CNN). Swin-T/S/B/L variants. Universal backbone for detection, segmentation, classification.

> **📊 Planned diagram:** Swin shifted window attention — two consecutive layers, showing W-MSA (regular windows) and SW-MSA (shifted windows).

> **📊 Planned diagram:** Swin vs. ViT feature hierarchy — Swin produces `[H/4, W/4]`, `[H/8, W/8]`, `[H/16, W/16]`, `[H/32, W/32]` feature maps for FPN.

## ConvNeXt: Modernizing CNNs (2022)

> **Planned content:** Take ResNet-50 and add every Swin/ViT design choice one by one. Result: ConvNeXt block — large kernel (7×7) depthwise conv + inverted bottleneck + LayerNorm + GELU + fewer activation/norm layers. Matches or beats Swin on all benchmarks. Strong argument for CNNs with modern training.

> **📊 Planned diagram:** ConvNeXt block vs. ResNet bottleneck — side-by-side showing the modernization changes.

## Architecture Comparison Table

> **Planned content:** Comprehensive comparison table of all major classification architectures.

> **📊 Planned table:** Architecture, year, ImageNet top-1 %, parameters (M), FLOPs (G), inference latency (ms on V100), training data needed.

**Next: [Chapter 19 — Detection Architectures →](./19_detection_architectures.md)**

---
*Last updated: May 2026*
