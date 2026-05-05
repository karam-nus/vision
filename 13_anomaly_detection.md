---
title: "Chapter 13 — Anomaly Detection"
---

[← Back to Table of Contents](./README.md)

# Chapter 13 — Anomaly Detection

> *"Train only on normal — detect everything else. Anomaly detection is the art of defining and defending normalcy."*

## Problem Formulation

> **Planned content:** One-class learning: only normal samples available during training. Detect deviations at inference. Three levels: image-level (is this image anomalous?), pixel-level (which pixels are anomalous?), and region-level. Industrial inspection, medical screening, fraud detection, surveillance.

> **📊 Planned diagram:** Anomaly detection taxonomy — image-level vs. pixel-level, and the three method families (reconstruction, embedding, normalizing flow).

## Reconstruction-Based Methods

> **Planned content:** Autoencoder (AE): train to reconstruct normal images. Anomalies → high reconstruction error. VAE: adds probabilistic latent space. Patch-level reconstruction. Limitation: AEs can also reconstruct anomalies well (the over-generalization problem). Memory-augmented AE (MemAE) to constrain the latent space.

> **📊 Planned diagram:** Autoencoder anomaly detection — normal image → AE → low error; anomalous image → AE → high error residual map.

## Embedding-Based Methods

> **Planned content:** PatchCore: extract patch features with ImageNet pre-trained ResNet, build a coreset (representative subset of training patch features), score test patches by distance to nearest neighbor in coreset. SPADE: KNN in feature space. Gaussian discriminant (Mahalanobis distance on features). No training beyond feature extraction.

> **📊 Planned diagram:** PatchCore pipeline — training: extract patch features `[N_patches, D]` → coreset subsampling → memory bank. Inference: test patch features → KNN distance → anomaly score map.

## Normalizing Flow Methods

> **Planned content:** Learn a bijective mapping from feature space to a standard Gaussian (base distribution). High likelihood → normal. Low likelihood → anomaly. FastFlow, CFLOW-AD. Advantages: exact likelihood, no AE over-generalization.

> **📊 Planned diagram:** Normalizing flow for anomaly detection — features → flow network → latent Gaussian → negative log-likelihood as anomaly score.

## Knowledge Distillation Methods

> **Planned content:** Student-teacher: teacher (ImageNet pretrained, fixed) and student (trained to mimic teacher on normals). At inference: discrepancy between teacher and student features is the anomaly score. RD4AD, SimpleNet. Efficient and competitive.

## Student-Teacher Feature Matching

> **Planned content:** Teacher network produces `[B, C, H/s, W/s]` feature maps. Student learns to produce identical features for normal inputs. Anomaly score: L2 distance between teacher and student feature maps.

## Industrial Inspection and MVTec Benchmark

> **Planned content:** MVTec AD dataset: 15 categories (textures + objects), 73 anomaly types. Evaluation: image-level AUROC, pixel-level AUROC, PRO (Per-Region Overlap). Current SOTA ~99% image AUROC. VisA benchmark for multi-class setting. Real-world challenges: multi-class, few-shot, continual.

> **📊 Planned diagram:** MVTec AD sample categories — showing normal and anomalous samples for metal nut, leather, wood, etc.

**Next: [Chapter 14 — OCR & Document Understanding →](./14_ocr_and_document_understanding.md)**

---
*Last updated: May 2026*
