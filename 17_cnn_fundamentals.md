---
title: "Chapter 17 — CNN Fundamentals"
---

[← Back to Table of Contents](./README.md)

# Chapter 17 — CNN Fundamentals

> *"The convolutional neural network is the inductive bias that made deep learning work for images — and understanding it is understanding the field's foundation."*

## The Convolution Operation

> **Planned content:** 2D convolution: filter slides over input, element-wise multiply + sum. Input `[B, C_in, H, W]`, filter `[C_out, C_in, k_h, k_w]`, output `[B, C_out, H', W']`. Padding: valid (no pad), same (pad to maintain size). Stride: step size. Output size formula: `H' = (H + 2P - k_h) / S + 1`. Bias term. Parameter count: `C_out * C_in * k_h * k_w + C_out`.

> **📊 Planned diagram:** 2D convolution animation — single filter sliding over a 5×5 feature map with stride 1, showing element-wise multiply-accumulate.

> **📊 Planned diagram:** Output spatial size as a function of input size, kernel size, padding, and stride.

$$H' = \left\lfloor \frac{H + 2P - k_h}{S} \right\rfloor + 1, \qquad W' = \left\lfloor \frac{W + 2P - k_w}{S} \right\rfloor + 1$$

```python
import torch
import torch.nn as nn

# Standard 2D convolution
conv = nn.Conv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True
)
# Parameters: 128 * 64 * 3 * 3 + 128 = 73,856

x = torch.randn(8, 64, 56, 56)   # [B=8, C_in=64, H=56, W=56]
y = conv(x)                        # [B=8, C_out=128, H=56, W=56]  (same padding)
```

## Receptive Field

> **Planned content:** The receptive field of a unit: which input pixels influence its output. Grows with depth. Stack of 3×3 convs: RF after L layers = `2L + 1`. A 7×7 conv = 3 stacked 3×3 convs but 3× more parameters. Effective receptive field vs. theoretical RF. Why deep narrow networks > shallow wide networks for CV.

> **📊 Planned diagram:** Receptive field growth — visualization of which input pixels affect a single neuron at depth 1, 2, 3 with 3×3 conv.

## Depth-wise and Separable Convolutions

> **Planned content:** Standard conv: `C_out * C_in * k * k` params. Depthwise separable: factorize into depthwise (`C_in` groups, `C_in * k * k`) + pointwise (`C_out * C_in * 1 * 1`). ~8-9× fewer parameters and FLOPs for 3×3 conv. MobileNet's core building block.

> **📊 Planned diagram:** Depthwise separable convolution decomposition — standard vs. DW+PW, with parameter counts compared.

```python
# Depthwise separable convolution
dw = nn.Conv2d(64, 64, 3, padding=1, groups=64)   # depthwise: groups=C_in
pw = nn.Conv2d(64, 128, 1)                         # pointwise: 1×1 conv

x = torch.randn(8, 64, 56, 56)   # [8, 64, 56, 56]
x = dw(x)                          # [8, 64, 56, 56]  depthwise
x = pw(x)                          # [8, 128, 56, 56] pointwise
```

## Pooling Strategies

> **Planned content:** Max pooling: take max in each window. Average pooling. Global average pooling (GAP): `[B, C, H, W] → [B, C, 1, 1] → squeeze → [B, C]` — replaces FC layers. Why GAP over fully-connected reduces parameters and enables variable input sizes. Adaptive pooling.

> **📊 Planned diagram:** Global average pooling — `[B, C, H, W]` → spatial mean per channel → `[B, C]` — showing how it aggregates spatial information.

## Normalization Layers

> **Planned content:** Batch Normalization (BN): normalize over `[B, H, W]` per channel → stable training, fast convergence. Learnable γ and β. Training vs. inference behavior (running stats). Layer Normalization (LN): normalize over `[C, H, W]` per sample → better for transformers. Instance Normalization (IN): normalize over `[H, W]` per channel per sample → style transfer. Group Normalization (GN): normalize over groups of channels → small batch safe.

> **📊 Planned diagram:** Four normalization methods — which axes are normalized for each, illustrated as a 4D tensor `[B, C, H, W]` with colored normalization regions.

## Activation Functions

> **Planned content:** ReLU: simple, dead neuron problem. Leaky ReLU: small negative slope. PReLU: learned slope. ELU, SELU. GELU: smooth approximation, used in transformers. SiLU/Swish: self-gated, used in EfficientNet, Swin. Mish. Hard variants (HardSwish, HardSigmoid) for mobile.

> **📊 Planned diagram:** Activation function comparison curves — ReLU, GELU, SiLU, Mish on the same axes.

## Dilated (Atrous) Convolutions

> **Planned content:** (Covered in depth in Ch. 4 for segmentation context) — theoretical foundation here: dilation rate r inserts `r-1` zeros between filter elements. Effective kernel size: `k_eff = k + (k-1)(r-1)`. Maintains spatial resolution while expanding receptive field. Gridding artifact for large dilation rates.

## Inductive Biases of CNNs

> **Planned content:** Translation equivariance (not invariance). Weight sharing across spatial locations. Local connectivity. Spatial hierarchy. Why these biases are beneficial for natural images but restrictive for long-range dependencies. Comparison with ViT's weaker inductive biases (more data required).

> **📊 Planned diagram:** Translation equivariance visualization — same object at different positions produces same features (shifted).

**Next: [Chapter 18 — Classification Architectures →](./18_classification_architectures.md)**

---
*Last updated: May 2026*
