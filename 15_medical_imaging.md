---
title: "Chapter 15 — Medical Imaging"
---

[← Back to Table of Contents](./README.md)

# Chapter 15 — Medical Imaging

> *"In medical imaging, a single false negative can cost a life — the stakes are different from natural images."*

## Medical Imaging Modalities

> **Planned content:** X-ray (2D projection, fast, radiation), CT (3D volumetric, cross-sections, higher dose), MRI (3D, soft tissue contrast, no radiation, slow), Ultrasound (real-time, portable, operator-dependent), Pathology slides (WSI: gigapixel images). Each modality has unique characteristics for model design.

> **📊 Planned diagram:** Medical modality overview — X-ray `[H, W, 1]`, CT `[D, H, W]`, MRI `[D, H, W]`, histopathology WSI (multi-resolution pyramid), fundus image `[H, W, 3]`.

## Segmentation in Medical Imaging

> **Planned content:** U-Net: the dominant architecture — encoder-decoder with skip connections, designed for medical images with limited training data. V-Net: 3D U-Net for volumetric data. nnU-Net: self-configuring U-Net baseline. TransUNet, SwinUNETR: transformer-based. Challenges: thin structures, class imbalance, 3D data, limited annotations.

> **📊 Planned diagram:** U-Net architecture with tensor shapes — input `[B, 1, 572, 572]` → encoder contracting path → bottleneck `[B, 1024, 28, 28]` → decoder expanding path → output `[B, 2, 388, 388]`.

## Classification in Medical Imaging

> **Planned content:** CheXNet: DenseNet-121 for 14 chest X-ray findings. Diabetic retinopathy grading. Skin lesion classification (ISIC). Pathology patch classification. Multi-label, imbalanced datasets. Grad-CAM visualization for explainability.

## Detection and Localization

> **Planned content:** Lung nodule detection (LUNA16). Polyp detection (colonoscopy). Lesion detection. Challenges: variable sizes, very small objects, 3D bounding boxes. DETR and Mask R-CNN adapted for medical images.

## Domain Shift and Limited Annotations

> **Planned content:** Domain gap between scanners, institutions, and protocols. Transfer learning from ImageNet vs. medical pre-training. Semi-supervised learning (mean teacher, pseudolabels). Active learning for annotation efficiency. Self-supervised pre-training (DINO on histology).

## Foundation Models for Medical Imaging

> **Planned content:** MedSAM: fine-tuned SAM for medical image segmentation. SuPreM: universal pre-training across modalities. BioViL-T: chest X-ray + radiology reports. CONCH: histopathology foundation model. Zero-shot generalization across organs and modalities.

> **📊 Planned diagram:** Foundation model for medical imaging — pre-training on multi-modal data → fine-tuning on specific tasks → comparison with task-specific baselines.

## Evaluation in Medical AI

> **Planned content:** Dice Score (segmentation). Hausdorff Distance (boundary quality). AUC-ROC. Sensitivity and specificity. FROC (Free-Response ROC) for detection. Inter-rater agreement (Cohen's κ). Regulatory considerations (FDA clearance, CE marking).

**Next: [Chapter 16 — Multimodal Vision Tasks →](./16_multimodal_vision_tasks.md)**

---
*Last updated: May 2026*
