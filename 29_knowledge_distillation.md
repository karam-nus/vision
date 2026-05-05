---
title: "Chapter 29 — Knowledge Distillation"
---

[← Back to Table of Contents](./README.md)

# Chapter 29 — Knowledge Distillation

> *"Distillation is mentorship in machine learning — a small student learns from a large teacher's internal reasoning, not just its final answers."*

## Hinton's Knowledge Distillation (2014)

> **Planned content:** Train a small student network using soft labels from a large teacher. Soft labels: the teacher's full probability distribution (not just argmax). Temperature T smooths the distribution, revealing inter-class relationships. KD loss = CE with soft labels + CE with hard labels (weighted sum). Why soft labels help: they encode "dark knowledge" — how similar classes relate.

> **📊 Planned diagram:** KD training — teacher produces soft probabilities `[B, C]`, student is trained with both teacher soft targets and ground-truth one-hot labels. Temperature effect on softmax distribution.

$$\mathcal{L}_{KD} = \alpha \cdot \mathcal{L}_{CE}(p_{student}, p_{teacher}^T) + (1-\alpha) \cdot \mathcal{L}_{CE}(p_{student}, y_{hard})$$

## Feature Mimicking (FitNets)

> **Planned content:** Match intermediate feature maps, not just final predictions. Student feature `[B, C_s, H, W]` → adapter conv → `[B, C_t, H, W]` ≈ teacher feature. Norm-based loss, attention map transfer. Why feature-level matching provides richer supervisory signal than just logits.

> **📊 Planned diagram:** Feature mimicking — teacher intermediate features vs. student features with adaptation layer, L2/cosine loss alignment.

## Attention Transfer

> **Planned content:** AT: transfer attention maps (spatial activation statistics) from teacher to student. Attention map `A = ||F||_p` (p-norm across channels). The spatial locations the teacher "looks at" guide student learning. Zagoruyko & Komodakis (2017).

## DKD: Decoupled Knowledge Distillation (2022)

> **Planned content:** Decompose KD loss into: TCKD (Target Class KD) — information about the target class, and NCKD (Non-Target Class KD) — inter-class relationships. Show that TCKD and NCKD have contradictory effects and can be decoupled and balanced separately. Consistently outperforms vanilla KD.

## Task-Specific KD in Vision

> **Planned content:** Detection KD: Mimic proposal features, FPN features, classification + box distributions. LD (Localization Distillation): transfer box distribution from teacher. Segmentation KD: Structured feature alignment, affinity graph transfer. Pose KD: Heatmap soft label distillation.

> **📊 Planned diagram:** Detection distillation — teacher detector features at FPN level P3-P7 vs. student, with weighted feature alignment loss.

## Self-Distillation

> **Planned content:** Deep supervision: shallower exits learn from the final output. Born-Again Networks: a sequence of students each trained on the previous model. DeiT: use CNN teacher to train ViT student. DINO: self-distillation without labels.

## Online Distillation (Mutual Learning)

> **Planned content:** No pre-trained teacher. Multiple students train simultaneously, each learning from the others. DML (Deep Mutual Learning). Collaborative learning. More robust than offline KD in some settings.

**Next: [Chapter 30 — Neural Architecture Search →](./30_neural_architecture_search.md)**

---
*Last updated: May 2026*
