---
title: "Chapter 35 — Self-Supervised Learning"
---

[← Back to Table of Contents](./README.md)

# Chapter 35 — Self-Supervised Learning

> *"The most valuable commodity in CV is labels. Self-supervised learning is the art of learning without them."*

## Why Self-Supervised Learning?

> **Planned content:** Labeled data bottleneck: ImageNet has 1M labels, but the internet has 5 billion images. SSL: design pretext tasks where the supervisory signal comes from the data itself. Two families: contrastive (pull together similar views, push apart dissimilar) and non-contrastive (no negative pairs). Downstream transfer: SSL features often outperform supervised when fine-tuned.

> **📊 Planned diagram:** SSL philosophy — two augmented views of the same image should produce similar representations; SSL framework overview.

## Contrastive Learning: SimCLR (2020)

> **Planned content:** Two augmented views of each image. Encoder `f(·)` + projection head `g(·)`. NT-Xent loss: attract same-image pairs, repel different-image pairs within a mini-batch. Large batch (4096-8192) essential (more negatives). Strong augmentation: crop + color jitter + grayscale + Gaussian blur. Two-layer MLP head (discard after pre-training).

> **📊 Planned diagram:** SimCLR framework — image → two augmented views → encoder → projection head → NT-Xent loss. Batch of N → 2N views → N positive pairs + 2(N-1) negative pairs.

$$\mathcal{L}_{NT-Xent} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

## MoCo: Momentum Contrast (2020)

> **Planned content:** Key idea: maintain a large queue of negative keys from past batches. Momentum encoder: slowly updated copy of encoder (like DINO teacher) to keep key representations consistent. Decouples batch size from number of negatives (queue size = 65536). MoCo v2: add MLP head, strong augmentation. MoCo v3: ViT backbone, no queue needed with large batch.

> **📊 Planned diagram:** MoCo queue mechanism — current mini-batch + large queue of past negatives → InfoNCE loss. Momentum encoder updated slowly (α=0.999).

## BYOL: Bootstrap Your Own Latent (2020)

> **Planned content:** No negative pairs! Student predicts teacher's representations of augmented views. Teacher = EMA of student. Predictor MLP on student side only. How does it avoid collapse? The asymmetry (predictor + EMA) prevents trivial solutions. Requires careful hyperparameter tuning.

> **📊 Planned diagram:** BYOL architecture — online network (encoder + projector + predictor) vs. target network (EMA encoder + projector). Loss = cosine similarity between online prediction and target projection.

## SimSiam: Exploring Collapse Prevention (2021)

> **Planned content:** No negative pairs, no momentum encoder. Stop gradient on one branch only. Stop-gradient is the key mechanism preventing collapse. VICReg: variance-invariance-covariance regularization. BarlowTwins: off-diagonal cross-correlation minimization.

## MAE: Masked Autoencoding (2021)

> **Planned content:** (Also covered in Ch. 24 for ViT context) — SSL framework: mask 75% of image patches, reconstruct masked patches with a lightweight decoder. Encoder only sees visible tokens (25%) — efficient. Decoder: 8-layer transformer, discarded after pre-training. Pixel-level MSE reconstruction loss. MAE pre-training on ImageNet-1K gives better fine-tuning than supervised on ImageNet-21K.

> **📊 Planned diagram:** MAE pre-training — 75% masked patches, encoder on 25% visible `[B, 49, 768]` → add mask tokens → decoder `[B, 196, 768]` → reconstruct `[B, 196, 768]` pixel values → MSE.

## DINO: Self-Distillation (2021)

> **Planned content:** (Also covered in Ch. 24) — SSL framework details: global and local crops. Student sees small crops (local), teacher sees large crops (global). This forces student to understand local crops using global context from teacher. Centering prevents collapse (subtract running mean from teacher output). Sharpening (low temperature for teacher).

## DINOv2: Large-Scale SSL (2023)

> **Planned content:** Scale DINO with: curated dataset (LVD-142M), iBOT (masked image modeling + DINO), SwAV multi-crop, KoLeo regularizer (entropy of pairwise distances). Registers: dedicated register tokens absorb global background information. DINOv2 features are directly usable as embeddings for segmentation, depth, classification without fine-tuning.

> **📊 Planned diagram:** DINOv2 training objectives — DINO global/local crop loss + iBOT masked patch reconstruction + KoLeo entropy regularizer combined.

## Transfer Learning from SSL Features

> **Planned content:** Linear probing: freeze backbone, train only linear head. Few-shot probing: k-NN classification. Full fine-tuning. SSL → supervised comparison: SSL often wins with >10M pre-training images. Task-specific benefits: DINOv2 for segmentation, MAE for detection, SimCLR for metric learning.

**Next: [Chapter 36 — Deployment →](./36_deployment.md)**

---
*Last updated: May 2026*
