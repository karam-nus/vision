---
title: "Appendix A — History of Computer Vision"
---

[← Back to Table of Contents](./README.md)

# Appendix A — History of Computer Vision

> *"To understand where computer vision is going, you must understand where it came from — and the surprising number of times the field has reinvented itself."*

## Pre-Deep-Learning Era (1959–2011)

### The Neurobiological Origins (1959–1970s)
> **Planned content:** Hubel & Wiesel's Nobel Prize work (1981) on simple and complex cells in the cat visual cortex — the biological basis for edge detectors and hierarchical feature processing. Block world (Larry Roberts, 1963): first CV thesis. DARPA autonomous vehicle (1970s). The "Summer Project" (1966) that thought CV would be solved by the end of the summer.

### Feature Descriptor Era (1980–2011)
> **Planned content:** SIFT (Lowe, 2004): scale-invariant feature transform, robust keypoint matching. SURF: faster SIFT. HOG (Dalal & Triggs, 2005): histograms of oriented gradients for pedestrian detection. Viola-Jones face detector (2001): Haar features + AdaBoost + integral images — first real-time face detection. Deformable Part Models (DPM, Felzenszwalb, 2008). PASCAL VOC challenge beginning (2005).

> **📊 Planned diagram:** SIFT keypoint detection and description pipeline — scale-space extrema → orientation assignment → 128D descriptor.

### Shallow Machine Learning for CV (1990s–2011)
> **Planned content:** SVM on HOG/SIFT features. Bag of Visual Words (BoVW): VQ of SIFT → histogram → SVM. Fisher Vector: improved BoVW encoding. Principal Component Analysis (PCA) for face recognition (Eigenfaces, Turk & Pentland, 1991). The kernel trick for non-linear CV.

## The Deep Learning Revolution (2012–2015)

### AlexNet and ImageNet (2012)
> **Planned content:** Krizhevsky, Sutskever, Hinton. 8-layer CNN, ReLU, dropout, GPU training. 15.3% top-5 error vs. 26.2% for runner-up. Changed everything. The role of ImageNet challenge in catalyzing progress. GPU training being the key enabler.

### The Feature Learning Explosion (2013–2015)
> **Planned content:** ZFNet (2013): visualizing CNN features. VGGNet (2014): 3×3 convolutions everywhere. GoogLeNet/Inception (2014): wider network, 1×1 convolutions, auxiliary classifiers. Batch Normalization (2015). R-CNN (2013) → Fast R-CNN (2015) → Faster R-CNN (2015): detection revolution.

## The Architecture Era (2015–2019)

> **📊 Planned diagram (timeline):** Architecture evolution — ResNet (2015) → DenseNet (2016) → MobileNet (2017) → NASNet (2018) → EfficientNet (2019).

### ResNet (2015)
> **Planned content:** He et al. Skip connections solving vanishing gradients. Won ILSVRC with 3.57% error. 152 layers — deeper than any prior network. Highway networks as predecessor.

### Semantic Segmentation Breakthrough (2015–2016)
> **Planned content:** FCN (2014 arxiv, 2015 CVPR): first end-to-end segmentation. U-Net (2015): medical imaging watershed. DeepLab v2 (2016): ASPP.

### Object Detection Explosion (2015–2019)
> **Planned content:** YOLOv1 (2015): detection as regression, 45 FPS. SSD (2016): multi-scale. FPN (2017): feature pyramids. Mask R-CNN (2017): instance segmentation. RetinaNet (2017): focal loss. YOLOv3 (2018). CornerNet (2018). FCOS (2019).

## The Transformer Era (2020–2023)

### Vision Transformer (2020)
> **Planned content:** Dosovitskiy et al. ViT: applying BERT-style transformers to image patches. Requires large data (JFT-300M). DeiT: ImageNet-only training. Swin (2021): hierarchical windows, SOTA on all dense prediction tasks.

### Self-Supervised Learning Revolution (2020–2022)
> **Planned content:** MoCo (2020), SimCLR (2020), BYOL (2020), MAE (2021), DINO (2021). Labels no longer required for good features. Transformers + SSL = new paradigm.

### CLIP and Multimodal Era (2021)
> **Planned content:** CLIP (OpenAI, 2021): contrastive language-image pre-training. Zero-shot classification. Foundation for all modern VLMs. DALL-E, GLIDE, DALL-E 2, Stable Diffusion.

## The Foundation Model Era (2023–Present)

> **Planned content:** SAM (2023): segment anything. DINOv2 (2023): universal visual features. GPT-4V (2023): multimodal LLM. Gemini 1.5 (2024): native multimodal. SAM 2 (2024): video segmentation. Florence-2 (2024): unified vision. The convergence of vision and language.

> **📊 Planned diagram (full timeline):** Comprehensive CV history timeline — key papers, paradigm shifts, and benchmark milestones from 1959 to 2026.

**[← Back to Table of Contents](./README.md)**

---
*Last updated: May 2026*
