---
title: "Chapter 2 — Image Classification"
---

[← Back to Table of Contents](./README.md)

# Chapter 2 — Image Classification

> *"Image classification is the 'hello world' of computer vision — deceptively simple to implement, surprisingly hard to master."*

## The Classification Problem

> **Planned content:** Mapping an image tensor `[B, C, H, W]` to a class label (or distribution over labels). Single-label (one class per image) vs. multi-label (multiple classes). Closed-set (fixed vocabulary) vs. open-set (novel classes at test time). Formal problem definition, loss formulation.

> **📊 Planned diagram (flowchart):** Classification pipeline — image → backbone → global average pooling → linear head → softmax → class probabilities → argmax → label.

## Softmax and Cross-Entropy Loss

> **Planned content:** Softmax function — converting raw logits to probabilities. Numerical stability (log-sum-exp trick). Cross-entropy loss derivation. Why cross-entropy = NLL of a categorical distribution. Label smoothing and why it helps. Temperature scaling for calibration. Class imbalance: weighted cross-entropy, focal loss in classification.

> **📊 Planned diagram:** Softmax geometry — how logits are transformed into a probability simplex. Visualization of CE loss surface.

$$\mathcal{L}_{CE} = -\sum_{c=1}^{C} y_c \log \hat{p}_c \quad \text{where} \quad \hat{p}_c = \frac{e^{z_c}}{\sum_{j} e^{z_j}}$$

```python
import torch
import torch.nn.functional as F

logits = torch.randn(4, 1000)  # [B=4, C=1000] — raw classifier output
labels = torch.randint(0, 1000, (4,))  # [B=4] — integer class indices

loss = F.cross_entropy(logits, labels)  # scalar — combines log_softmax + NLL
probs = F.softmax(logits, dim=-1)       # [B, C] — sum to 1 per sample
top5 = probs.topk(5, dim=-1)           # (values [B,5], indices [B,5])
```

## Training Pipeline

> **Planned content:** Full training loop. Data loading and augmentation. Mixed precision training (`torch.autocast`). Learning rate scheduling. Validation loop. Model checkpointing. Transfer learning from ImageNet pre-trained weights.

> **📊 Planned diagram (flowchart):** Full training loop — data loader → augment → forward pass → loss → backward → optimizer step → scheduler step → validation → checkpoint.

## Evaluation Metrics

> **Planned content:** Top-1 accuracy. Top-5 accuracy (ImageNet standard). Per-class accuracy and macro-average. Confusion matrix analysis. Calibration metrics (ECE). Mean per-class accuracy for long-tailed datasets.

> **📊 Planned diagram:** Confusion matrix visualization with class recall and precision highlighted.

## Fine-Grained Recognition

> **Planned content:** When classes are very similar (dog breeds, bird species, car models). The role of discriminative parts. Bilinear pooling. Part-based attention models. Datasets: CUB-200, Stanford Cars, iNaturalist.

> **📊 Planned diagram:** Fine-grained recognition challenge — showing subtle inter-class differences at the feature level.

## Multi-Label Classification

> **Planned content:** Multiple classes present simultaneously (e.g., COCO multi-label). Binary cross-entropy (BCE) per class. Sigmoid instead of softmax. Handling class co-occurrence. Asymmetric loss (ASL) for handling easy negatives.

```python
# Multi-label: sigmoid + binary cross-entropy
logits = torch.randn(4, 80)    # [B=4, C=80] — one logit per class
targets = torch.randint(0, 2, (4, 80)).float()  # [B=4, C=80] — binary labels

loss = F.binary_cross_entropy_with_logits(logits, targets)  # scalar
probs = torch.sigmoid(logits)   # [B=4, C=80] — independent probabilities
preds = (probs > 0.5).float()  # [B=4, C=80] — binary predictions
```

## Zero-Shot and Few-Shot Classification

> **Planned content:** Zero-shot: classify images into classes never seen during training (CLIP-based approaches). Few-shot: 1-shot, 5-shot learning. Prototypical networks. Meta-learning basics. Prompt engineering for zero-shot classification.

> **📊 Planned diagram:** CLIP zero-shot classification — image embedding vs. text embedding cosine similarity.

## Long-Tail Distribution Problem

> **Planned content:** Real-world class frequencies follow a power law. Head classes (thousands of samples) vs. tail classes (< 10 samples). Strategies: re-sampling, re-weighting, two-stage training, logit adjustment. iNaturalist challenge.

> **📊 Planned diagram:** Long-tail distribution curve — class frequency vs. class rank, head vs. tail regions.

## Practical Code: End-to-End Classification

> **Planned content:** Full working example — loading a pre-trained ResNet-50, fine-tuning on a custom dataset, evaluating, and running inference. Includes data loading, transforms, training loop, and saving results.

> **📊 Planned code block:** Complete training script with comments showing tensor shapes at every step.

## Benchmark: ImageNet

> **Planned content:** ILSVRC challenge history. The 1000-class benchmark. Train/val/test splits. The surprise of 2012 (AlexNet). Progress from 56% to 91%+ top-1 accuracy. Saturation and what comes next.

> **📊 Planned diagram:** ImageNet top-1 accuracy over time — bar chart from AlexNet (2012) to modern ViT/ConvNeXt.

**Next: [Chapter 3 — Object Detection →](./03_object_detection.md)**

---
*Last updated: May 2026*
