---
title: "Chapter 30 — Neural Architecture Search"
---

[← Back to Table of Contents](./README.md)

# Chapter 30 — Neural Architecture Search

> *"If human intuition designed ResNet and EfficientNet, imagine what an algorithm optimizing for your specific hardware and dataset could find."*

## Manual vs. Automated Architecture Design

> **Planned content:** Manual design: domain experts iterating on intuitions (VGG → ResNet → DenseNet). Time-consuming, biased toward human-intuitive structures. NAS: define a search space, search strategy, and performance estimator → automate the architecture design process.

> **📊 Planned diagram:** NAS framework — search space (building blocks, connections) + search strategy (controller, evolutionary, gradient) + performance estimation (training, proxy) → optimal architecture.

## Search Strategies

> **Planned content:** Reinforcement Learning (NASNet, MnasNet): RNN controller generates architectures, reward = validation accuracy. Slow (GPU-years). Evolutionary (AmoebaNet): population of architectures, mutate and select best. DARTS: Differentiable Architecture Search — relax discrete choices to continuous (weighted sum of operations), optimize with gradient. One-shot NAS: train a supernet, derive subnetworks by weight sharing.

> **📊 Planned diagram:** DARTS mixed operation — edge in search graph = weighted sum of candidate operations (conv3×3, conv5×5, skip, etc.), weights learned by gradient.

## DARTS: Differentiable Architecture Search

> **Planned content:** Continuous relaxation: replace discrete architecture choices with softmax over operations. Bilevel optimization: alternately update architecture weights α (on validation set) and operation weights W (on training set). Derive final architecture by argmax of α. Limitations: instability, skip connection dominance.

$$\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o'} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x)$$

## MobileNetV3 and Platform-Aware NAS

> **Planned content:** MnasNet: optimize directly for on-device latency × accuracy. Look-up table for layer latency on target hardware. ProxylessNAS: directly optimize on target hardware with latency measured. Platform-aware: same NAS run on GPU vs. mobile gives different architectures.

## EfficientNet: NAS + Compound Scaling

> **Planned content:** (Architecture covered in Ch. 18) — NAS process here: MnasNet-style NAS to find EfficientNet-B0 baseline. Then compound scaling to derive B1-B7. The scaling coefficients α, β, γ found by grid search.

## Once-For-All: Train Once, Deploy Anywhere

> **Planned content:** Train a single supernet that supports many subnetworks of varying depth/width/kernel size/resolution. Progressive shrinking training: gradually reduce subnetwork size. At deployment: look up accuracy-latency table for your hardware, pick the right subnetwork. No retraining needed.

> **📊 Planned diagram:** Once-for-All supernet — single training produces a family of models at different efficiency points, deployed to phone/edge/server by choosing subnetwork.

## NAS for Detection and Segmentation

> **Planned content:** SpineNet: NAS backbone for detection with cross-scale connections. DetNAS: NAS for one-stage detector. SqueezeNAS for segmentation. Auto-DeepLab: NAS for semantic segmentation.

**Next: [Chapter 31 — Efficient Inference →](./31_efficient_inference.md)**

---
*Last updated: May 2026*
