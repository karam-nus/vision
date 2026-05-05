---
title: "Chapter 34 — Training Recipes"
---

[← Back to Table of Contents](./README.md)

# Chapter 34 — Training Recipes

> *"The difference between a 75% and an 80% model is often not the architecture — it's the training recipe."*

## Loss Functions in Computer Vision

> **Planned content:** Cross-entropy (classification). BCE with logits (multi-label, detection objectness). Focal loss (imbalanced detection). Smooth L1 / Huber (bounding box regression). IoU-based losses (GIoU, DIoU, CIoU for detection). Dice + Lovász (segmentation). Wing loss (landmark regression). OKS loss (pose estimation heatmaps: MSE on heatmaps). Triplet loss, ArcFace (metric learning).

> **📊 Planned table:** Loss function reference — task, formula, hyperparameters, when to use.

## Optimizers

> **Planned content:** SGD + momentum: classic, requires careful LR tuning. AdamW: adaptive LR, weight decay decoupled (Loshchilov & Hutter). Best for ViT/transformer training. LARS (Layer-Adaptive Rate Scaling): scales LR per layer based on weight norm, enables large-batch training. LAMB: LARS for transformers. SAM (Sharpness-Aware Minimization): seeks flatter minima, improved generalization. Lion (EvoLved Sign Momentum): sign-based update, memory efficient.

> **📊 Planned diagram:** SGD vs. AdamW loss landscape trajectory — SGD with sharp minimum vs. SAM's flat minimum, and why flat = better generalization.

## Learning Rate Schedules

> **Planned content:** Constant LR: simple but suboptimal. Step decay (every 30 epochs ÷10). Multi-step. Cosine annealing: smooth decay to 0. Cosine with restarts (SGDR). OneCycleLR: 1-cycle policy (warm up → peak → anneal). Warmup + cosine: standard for ViT training (5-20 epoch warmup). Linear warmup formula. Effect of warmup for batch-normalized models.

> **📊 Planned diagram:** LR schedule comparison — step decay vs. cosine annealing vs. OneCycle vs. warmup+cosine, all on same axes (epoch vs. LR).

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

# Warmup 5 epochs + cosine 295 epochs
warmup = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=295, eta_min=1e-5)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
```

## Exponential Moving Average (EMA)

> **Planned content:** Maintain a shadow copy of model weights as EMA: `θ_ema = α * θ_ema + (1-α) * θ`. EMA model used for validation/inference. Smoother loss surface. Used in YOLO, DINO, SAM. Typical α = 0.999. Why EMA helps: averages out gradient noise, approximates ensemble.

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.data = self.decay * ema_p.data + (1 - self.decay) * p.data
            # ema_p: exponential moving average of parameter
```

## Bag of Tricks for ResNet Training

> **Planned content:** He et al. "Bag of Tricks for Image Classification": Linear LR scaling rule (LR ∝ batch size / 256). Warmup. No bias decay. Label smoothing. MixUp. Cosine LR. ResNet-D: anti-aliased downsampling. Zero-initialization of last BN in residual blocks.

## Large-Batch Training

> **Planned content:** Linear scaling rule. LARS for very large batches (64K+). Gradient accumulation as a substitute. Synchronous vs. asynchronous data-parallel training. Batch size vs. accuracy at ImageNet.

## Gradient Clipping

> **Planned content:** Why gradients explode (especially in transformers, RNNs). Gradient norm clipping: rescale gradient if norm exceeds threshold. Gradient value clipping. PyTorch `nn.utils.clip_grad_norm_`. Typical threshold: 1.0 for transformers, higher for CNNs.

## Mixed Precision Training

> **Planned content:** Train with FP16/BF16, master weights in FP32. `torch.autocast` + `GradScaler`. Loss scaling to prevent underflow in FP16. BF16 advantages (larger range, no scaling needed). Memory saving: ~2× vs. FP32.

**Next: [Chapter 35 — Self-Supervised Learning →](./35_self_supervised_learning.md)**

---
*Last updated: May 2026*
