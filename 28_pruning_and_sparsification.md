---
title: "Chapter 28 — Pruning and Sparsification"
---

[← Back to Table of Contents](./README.md)

# Chapter 28 — Pruning and Sparsification

> *"Pruning is the art of removing what doesn't matter — and discovering that often, most of it doesn't."*

## Unstructured vs. Structured Pruning

> **Planned content:** Unstructured: remove individual weights (arbitrary sparsity pattern). High compression ratio but requires sparse hardware to see speedup. Structured: remove entire filters, channels, attention heads, or layers. Directly reduces FLOPs, hardware-agnostic. Semi-structured: 2:4 sparsity (2 out of every 4 weights = 0) — Ampere GPU sparse Tensor Cores.

> **📊 Planned diagram:** Three sparsity patterns — unstructured (random), structured (channel/filter), semi-structured (2:4) — showing weight matrix with zeros colored.

## Magnitude-Based Pruning

> **Planned content:** Simple heuristic: set smallest |w| weights to zero. Global threshold vs. per-layer threshold. One-shot vs. iterative pruning. Lottery Ticket Hypothesis: a sparse subnetwork ("winning ticket") that can be trained from scratch to same accuracy. Finding the ticket with IMP (Iterative Magnitude Pruning).

> **📊 Planned diagram:** Iterative magnitude pruning cycle — train → evaluate → prune p% smallest weights → rewind weights to initialization → repeat.

$$\text{mask}_i = \mathbf{1}\left[|w_i| \geq \tau\right], \qquad \tau = \text{percentile}(|w|, p)$$

## Gradual Magnitude Pruning (GMP)

> **Planned content:** Start dense, gradually increase sparsity during training. Cubic sparsity schedule: `s(t) = s_f(1 - (1 - t/T)^3)`. Fine-tuning with mask applied. Works better than one-shot for maintaining accuracy.

## Movement Pruning

> **Planned content:** Prune weights that are moving toward zero during fine-tuning (negative movement). More principled than magnitude for task-specific pruning. Top-V masking. Used in SparseGPT-style approaches.

## Structured Pruning: Channels and Filters

> **Planned content:** Prune least important output filters of Conv2d. Importance criteria: L1 norm, Taylor expansion (gradient × activation). Network Slimming: learn BN scale factors, prune channels with small γ. Hard to do post-hoc without accuracy recovery — requires fine-tuning.

> **📊 Planned diagram:** Channel pruning — remove output channels with lowest importance score, adjusting next layer's input channels correspondingly.

## Ampere 2:4 Sparsification (Semi-Structured)

> **Planned content:** NVIDIA's Sparse Tensor Core acceleration: exactly 2 of every 4 consecutive weights must be zero. 2× speedup for matmul. ASP (Automatic Sparsity): prune to 2:4 pattern + fine-tune. Part of PyTorch sparse training tools.

> **📊 Planned diagram:** 2:4 structured sparsity — 4-element window with 2 non-zeros, hardware compresses weight matrix 2× and accelerates accordingly.

## Network Architecture vs. Pruning

> **Planned content:** Liu et al. finding: pruned architecture + random re-initialization ≥ pruned architecture + original weights. Suggests structured pruning is really NAS. Practical implication: just design a smaller network with NAS or MobileNet.

**Next: [Chapter 29 — Knowledge Distillation →](./29_knowledge_distillation.md)**

---
*Last updated: May 2026*
