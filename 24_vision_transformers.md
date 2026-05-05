---
title: "Chapter 24 — Vision Transformers"
---

[← Back to Table of Contents](./README.md)

# Chapter 24 — Vision Transformers

> *"The transformer's success in language asks a natural question: can attention replace convolution? The answer — yes, given enough data — reshaped computer vision."*

## Adapting Transformers to Images

> **Planned content:** Why naive application of self-attention to images fails: for a 224×224 image, N = 50,176 tokens → O(N²) attention is 2.5B operations. The patch trick: divide image into non-overlapping patches, treat each as a token. Reduces N to 196 for 16×16 patches.

> **📊 Planned diagram:** Image → patch grid decomposition — `[3, 224, 224]` divided into 196 patches of `[3, 16, 16]` each, flattened and projected to `[196, 768]`.

## ViT Deep Dive

> **Planned content:** Patch embedding (linear projection of flattened patches). CLS token. Learnable position embeddings (1D, how they work better than 2D in practice). Multi-head self-attention across all patch tokens. MLP block. Layer normalization (pre-norm for stability). Classification via CLS token. ViT-S/B/L/H variants.

> **📊 Planned diagram:** ViT block — LayerNorm → MHSA → residual → LayerNorm → MLP → residual. Tensor shapes: `[B, N+1, D]` throughout (N=196 patches + 1 CLS token).

> **📊 Planned diagram:** Attention maps in ViT — visualizing attention from CLS token to patches, showing semantic segmentation-like behavior in late layers.

```python
# ViT attention visualization
model = timm.create_model('vit_base_patch16_224', pretrained=True)
# Last layer attention: [B, num_heads, N+1, N+1]
# Attention from CLS to patches: [B, num_heads, 1, N]
# Average over heads → [B, N] → reshape → [B, 14, 14]
```

## DeiT: Training ViTs Efficiently (2020)

> **Planned content:** Data-Efficient Image Transformer: train on ImageNet-1K only (no JFT). Knowledge distillation from a CNN teacher via a special distillation token. Hard distillation label = teacher's argmax. DeiT-S and DeiT-B competitive with ResNet. Key training tricks: strong augmentation (RandAugment, CutMix, Mixup), stochastic depth.

> **📊 Planned diagram:** DeiT — CLS token + distillation token + N patch tokens, showing separate classification and distillation prediction heads.

## Swin Transformer Deep Dive (2021)

> **Planned content:** Four-stage hierarchical architecture. Stage 1: `[B, H/4, W/4, 96]` (patch partition + linear embed). Stages 2-4: patch merging halves resolution, doubles channels. Window multi-head self-attention (W-MSA): each window `7×7` patches, attention within window only. Shifted window (SW-MSA): cyclic shift + masking for cross-window communication in odd layers. Relative position bias.

> **📊 Planned diagram:** Swin window partition and shift — regular windows (W-MSA) vs. shifted windows (SW-MSA), showing cyclic shift and masking for valid attention.

> **📊 Planned diagram:** Swin feature map sizes — `[H/4, W/4]` → `[H/8, W/8]` → `[H/16, W/16]` → `[H/32, W/32]` like a 4-stage CNN.

## MAE: Masked Autoencoding (2021)

> **Planned content:** Self-supervised pre-training: mask 75% of patches, reconstruct masked pixels with decoder. Sparse ViT encoder (only 25% visible tokens). Lightweight decoder. Pre-training is fast. Fine-tuning reaches ViT performance with far less labeled data. Why high masking ratio works (images are redundant, unlike text).

> **📊 Planned diagram:** MAE pipeline — mask 75% of patches → encoder on visible 25% `[B, 0.25*N, D]` → add mask tokens → decoder → reconstruct masked pixels → MSE loss.

## DINO: Self-Distillation with No Labels (2021)

> **Planned content:** Self-supervised training via self-distillation. Student network: sees distorted crops. Teacher network: EMA of student. Loss: cross-entropy between student and teacher output (soft labels over prototypes). No labels. Centering to prevent collapse. Local-to-global correspondence.

> **📊 Planned diagram:** DINO training — student (random augment) vs. teacher (large crop EMA) with centering and sharpening.

## DINOv2: Better Self-Supervised Features (2023)

> **Planned content:** Large-scale curated data (LVD-142M). DINO + iBOT (masked image modeling) combined. Registers for global/background understanding. Strong zero-shot features: k-NN classification, segmentation, depth, without task-specific fine-tuning. The "visual GPT-3" analogy.

> **📊 Planned diagram:** DINOv2 feature quality — PCA of patch features showing object part segmentation emerging from self-supervised training alone.

## EVA, InternImage, and Beyond

> **Planned content:** EVA-CLIP: large-scale CLIP pre-training for ViT-18B. InternImage: deformable convolution as a vision foundation model. Scaling laws for vision transformers. SoViT. FlexiViT. NaViT (variable resolution). PatchDropout.

**Next: [Chapter 25 — Multimodal Architectures →](./25_multimodal_architectures.md)**

---
*Last updated: May 2026*
