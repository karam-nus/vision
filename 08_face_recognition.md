---
title: "Chapter 8 — Face Recognition"
---

[← Back to Table of Contents](./README.md)

# Chapter 8 — Face Recognition

> *"A face is not a class — it is an identity. The goal is not classification but discrimination: to tell apart 7 billion unique faces."*

## The Face Recognition Pipeline

> **Planned content:** Four-stage pipeline: face detection → face alignment → feature extraction → metric-based matching. Each stage is critical. Detection failure = pipeline failure. Poor alignment = degraded features. Weak backbone = indistinguishable embeddings. Wrong metric = poor threshold calibration.

> **📊 Planned diagram (flowchart):** Full face recognition pipeline — raw image → face detection (bounding boxes + landmarks) → affine alignment → normalized face crop `[1, 3, 112, 112]` → backbone → embedding `[1, 512]` → cosine similarity with gallery → match/no-match decision.

## Face Detection

> **Planned content:** Challenges unique to faces: extreme scale variation, partial occlusion, non-frontal poses, expression, lighting. MTCNN: cascaded P-Net, R-Net, O-Net with landmark regression. RetinaFace: FPN-based multi-task (classification, regression, 5-landmark). SCRFD: efficient sample-constrained face detection. BlazeFace: mobile-optimized.

> **📊 Planned diagram:** MTCNN cascade — P-Net (fast reject at stride 2, output `[H/2, W/2, 2]`), R-Net (refine candidates), O-Net (precise localization + 5 landmarks).

## Face Alignment

> **Planned content:** Why alignment matters: a 5-degree rotation can significantly hurt recognition accuracy. 5-point landmark alignment (left/right eye center, nose tip, mouth corners). Similarity transform to a canonical face template. The canonical coordinates (e.g., ArcFace's 112×112 template). Affine vs. similarity transform.

> **📊 Planned diagram:** Alignment process — detected face with 5 landmarks → similarity transform → normalized 112×112 face crop, showing transformation matrix.

```python
import cv2
import numpy as np

# Standard ArcFace/CosFace reference template (5-point, 112x112)
REF_LANDMARKS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose tip
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.2041],  # right mouth corner
], dtype=np.float32)

def align_face(image, landmarks, output_size=(112, 112)):
    """
    image:     [H, W, 3]  — original image
    landmarks: [5, 2]     — detected 5-point landmarks
    Returns:   [112, 112, 3] — aligned face crop
    """
    M, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32),
        REF_LANDMARKS * (output_size[0] / 112.0),
        method=cv2.LMEDS
    )
    return cv2.warpAffine(image, M, output_size)  # [112, 112, 3]
```

## Feature Extraction Backbone

> **Planned content:** IResNet (Improved ResNet) — standard backbone for face recognition. MobileFaceNet — efficient mobile backbone. PartialFC trick for large-scale identity classes. The face embedding `[B, 512]`. Normalization: L2-normalize embeddings before the loss layer.

> **📊 Planned diagram:** Face backbone architecture — input `[B, 3, 112, 112]` → IResNet-50 → global average pool → FC → L2 normalize → embedding `[B, 512]`.

## ⭐ Metric Learning for Face Recognition — Deep Dive

> **Planned content:** The fundamental challenge: millions of training identities, but only a few samples per identity. Cannot use standard softmax (too many classes, doesn't generalize to unseen identities). Need embeddings where same-identity is close and different-identity is far. The journey from FaceNet (triplet loss) to ArcFace.

> **📊 Planned diagram:** Embedding space visualization — 2D projection of face embeddings, showing intra-class compactness and inter-class margin.

### Triplet Loss (FaceNet)

> **Planned content:** Anchor-Positive-Negative triplets. Triplet loss: `max(d(a,p) - d(a,n) + margin, 0)`. Online hard mining: semi-hard and hard negatives. Collapse failure. Mining difficulty as training progresses. FaceNet's 200M+ training data requirement.

> **📊 Planned diagram:** Triplet loss geometry — anchor, positive (same identity), negative (different identity) in embedding space, with margin illustrated.

### Softmax and Its Limitations

> **Planned content:** Standard softmax: `W^T x + b → cross-entropy`. The weight matrix has one column per identity. Works for closed-set but embedding quality is poor for open-set. The inter-class angular distribution is not well-controlled.

### SphereFace (A-Softmax)

> **Planned content:** Project embeddings onto a hypersphere (normalize both W and x). Replace inner product with cosine similarity. A-Softmax: add multiplicative angular margin m. The manifold of face space is better modeled on a hypersphere. Training instability issues.

### CosFace (LMCL)

> **Planned content:** Additive cosine margin (subtract m from cosine of target class). `cos(θ) - m`. Simpler and more stable than SphereFace. Scale factor s to control softmax temperature.

$$\mathcal{L}_{CosFace} = -\log \frac{e^{s(\cos\theta_{y_i} - m)}}{e^{s(\cos\theta_{y_i} - m)} + \sum_{j \neq y_i} e^{s \cos\theta_j}}$$

### ⭐ ArcFace — Deep Dive

> **Planned content:** The most widely used face recognition loss. Additive **angular** margin (not cosine margin): penalize in the angle domain. `cos(θ + m)` for the target class. Why angular margin is geometrically more principled than cosine margin (uniform margin on the hypersphere). The additive angular margin gives cleaner decision boundaries. Derivation step by step.

> **📊 Planned diagram:** ArcFace geometry on unit hypersphere — showing the decision boundary, angular margin m, and how different loss functions (Softmax, CosFace, ArcFace) place their margins.

> **📊 Planned diagram (flowchart):** ArcFace forward pass — embedding `[B, 512]` → L2 normalize → `[B, 512]` unit vector → dot with normalized weight matrix `[512, C]` → `[B, C]` cosines → `arccos` of target class → add margin m → `cos` back → scale s → softmax + cross-entropy.

$$\theta_{y_i} = \arccos(W_{y_i}^T x_i), \qquad \mathcal{L}_{ArcFace} = -\log \frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cos\theta_j}}$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceHead(nn.Module):
    """ArcFace (Additive Angular Margin) loss head."""
    def __init__(self, in_features, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s = s  # scale factor
        self.m = m  # angular margin (radians)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)   # cos(m)
        self.sin_m = math.sin(m)   # sin(m)
        self.th    = math.cos(math.pi - m)  # cos(π - m)
        self.mm    = math.sin(math.pi - m) * m  # sin(π - m) * m

    def forward(self, x, labels):
        """
        x:      [B, in_features]  — L2-normalized embeddings
        labels: [B]               — identity class indices
        Returns: logits [B, num_classes] for cross-entropy
        """
        # Normalize weight matrix columns
        W = F.normalize(self.weight, dim=1)  # [C, in_features]

        # Cosine similarity: cos(θ)
        cosine = F.linear(F.normalize(x), W)  # [B, C]

        # Compute cos(θ + m) for target class
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))  # [B, C]
        phi  = cosine * self.cos_m - sine * self.sin_m       # [B, C]  cos(θ+m)

        # Stable fallback for θ > π - m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # [B, C]

        # One-hot: apply margin only to target class
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()  # [B, C]
        logits  = (one_hot * phi) + ((1.0 - one_hot) * cosine)          # [B, C]
        logits  = logits * self.s  # scale

        return logits  # [B, C]  — pass to cross-entropy
```

### AdaFace

> **Planned content:** Adaptive margin based on image quality. High-quality images get larger margins. Low-quality images get smaller margins. Norm of embedding as proxy for quality. Why this improves in-the-wild recognition.

## Open-Set vs. Closed-Set Recognition

> **Planned content:** Closed-set: test identities appear in training (classification). Open-set: test identities are new (verification). 1:1 verification: is this the same person? 1:N identification: who is this person? FAR/FRR/TAR@FAR metrics. Setting the decision threshold.

> **📊 Planned diagram:** ROC curve for face verification — TAR vs. FAR, with operating points at FAR=1e-4, 1e-3.

## Anti-Spoofing (Liveness Detection)

> **Planned content:** Print attack, replay attack, 3D mask attack. Binary classification: real vs. spoof. Depth-based methods. Micro-texture analysis. Temporal consistency. Datasets: NUAA, MSU-MFSD, SiW.

## Evaluation Benchmarks

> **Planned content:** LFW (Labeled Faces in the Wild): 6000 pairs, saturated. IJB-B/C: harder, unconstrained. MegaFace: 1M distractor gallery. MS-Celeb-1M training set. AgeDB, CFP-FP, CPLFW, CALFW for specific attributes.

**Next: [Chapter 9 — Depth Estimation →](./09_depth_estimation.md)**

---
*Last updated: May 2026*
