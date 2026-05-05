---
title: "Chapter 32 — Datasets"
---

[← Back to Table of Contents](./README.md)

# Chapter 32 — Datasets

> *"In CV, you are what you train on. The dataset defines the task, the difficulty, and the generalization bound."*

## Image Classification Datasets

> **Planned content:** CIFAR-10/100 (60K images, 10/100 classes, 32×32). ImageNet-1K (1.28M train, 50K val, 1000 classes, ~469px avg). ImageNet-21K (14M images, 21K classes). iNaturalist (species-level, long tail, 675K images). Food-101, Stanford Cars, CUB-200. Oxford 102 Flowers. How to use ImageNet pre-trained models.

> **📊 Planned table:** Classification datasets — images, classes, resolution, split sizes, typical top-1 accuracy of SOTA.

## Object Detection Datasets

> **Planned content:** PASCAL VOC (2007/2012): 20 classes, 11K train/5K test, mAP@50. COCO (2017): 80 classes, 118K train/5K val, 20K test, hierarchical categories, 1.5M instances, stuff+things. Open Images v7: 9M images, 600 classes, relationship annotations. LVIS: long-tail COCO extension, 1203 categories. Objects365: 365 classes, 600K images (large-scale pre-training). How COCO annotations are structured (JSON format).

> **📊 Planned diagram:** COCO annotation JSON structure — image_info, annotations (bbox xyxy, category_id, segmentation polygon, keypoints), categories.

## Segmentation Datasets

> **Planned content:** PASCAL VOC segmentation (21 classes including background). ADE20K: 150 semantic categories, 20K training images, scene parsing. Cityscapes: urban driving scenes, 19 classes, 5K fine + 20K coarse. COCO-Stuff: 91 stuff classes + 80 thing classes (panoptic). SA-1B: SAM's 11M images + 1B masks (largest segmentation dataset). Mapillary Vistas.

## Pose Estimation Datasets

> **Planned content:** MPII Human Pose: 25K images, 40K persons, 16 keypoints, mostly single-person. COCO Keypoints: 200K images, 250K persons, 17 keypoints, multi-person. Human3.6M: 3.6M frames, 3D ground truth from motion capture, 11 subjects. MPI-INF-3DHP: in-the-wild 3D pose. COCO WholeBody: 133 keypoints (body+face+hand+foot). PoseTrack: video pose + tracking.

## Face Datasets

> **Planned content:** LFW (Labeled Faces in the Wild): 13K images, 5749 identities, 6K verification pairs. MS-Celeb-1M training: 100K identities, 10M images (now removed). WebFace260M (current large-scale). IJB-B/C: unconstrained, 1845/3531 subjects. AgeDB: 16K images, age-varied pairs. CFP-FP: frontal-profile pairs. WIDER FACE: 393K faces for detection.

## Anomaly Detection Datasets

> **Planned content:** MVTec AD: 5354 train (normal only) + 1725 test (normal + anomaly), 15 categories. VisA: 10,821 images, 12 objects, multi-class. MPDD (metal parts). DAGM (texture). BeanTech Anomaly Detection Dataset (BTAD).

## Depth and 3D Datasets

> **Planned content:** KITTI: 7481 stereo training images + LiDAR, outdoor driving. NYU-Depth v2: 1449 RGB-D indoor frames, 40 classes. ScanNet: 2.5M frames, 1513 scenes, 3D meshes. 7-Scenes: RGB-D relocalization. Waymo Open Dataset: 200K LiDAR scenes. nuScenes: 1000 driving scenes, 6 cameras + LiDAR + radar.

## Video Datasets

> **Planned content:** Kinetics-400/600/700: action recognition, 10s YouTube clips. Something-Something v2: 220K clips, temporal reasoning (174 classes). UCF-101: 13K clips, 101 action classes. HMDB-51. AVA: 430 15min clips, spatio-temporal action annotation. DAVIS: 90 high-quality video object segmentation sequences. MOT17/MOT20: multi-object tracking benchmarks. DanceTrack, SportsMOT.

## OCR and Document Datasets

> **Planned content:** ICDAR 2015/2017/2019: scene text detection and recognition competitions. TextOCR: 900K annotated words from Open Images. Total-Text: curved text. FUNSD: form understanding. DocVQA: document visual question answering. PubLayNet: 1M scientific document pages.

## Medical Imaging Datasets

> **Planned content:** ChestX-ray14: 112K X-rays, 14 pathology labels. MIMIC-CXR: 227K X-rays + radiology reports. BraTS: multimodal brain tumor segmentation. LIDC-IDRI: lung nodule detection. ISIC: skin lesion classification. Camelyon16/17: computational pathology (lymph node metastases).

**Next: [Chapter 33 — Data Augmentation →](./33_data_augmentation.md)**

---
*Last updated: May 2026*
