---
title: "Appendix B — Math Foundations for Computer Vision"
---

[← Back to Table of Contents](./README.md)

# Appendix B — Math Foundations for Computer Vision

> *"You don't need to re-derive everything — but you need to know enough to read a paper and debug a model."*

## Linear Algebra for CV

> **Planned content:** Vectors and norms. Matrix multiplication: `[M, K] × [K, N] → [M, N]`. Dot product as similarity. Cosine similarity. Outer product. Transpose. Inverse (and when it doesn't exist). Eigendecomposition: `Av = λv`. SVD: `A = UΣVᵀ` — PCA, low-rank approximation. Rank. Determinant. Trace. Why these matter: attention is matrix multiplication, PCA is SVD, cosine similarity is used everywhere.

> **📊 Planned diagram:** Matrix multiplication visualization — `[M, K] × [K, N]` as dot products, showing row × column = element.

## Convolution as Matrix Operations

> **Planned content:** 2D convolution expressed as matrix multiplication (im2col trick): rearrange input patches into rows of a matrix, stack filter weights as rows → one matrix multiplication. Why PyTorch uses im2col + GEMM under the hood. Relationship to cross-correlation (convolution without flip). Fourier transform of convolution = element-wise multiplication.

> **📊 Planned diagram:** im2col transformation — how a `3×3` sliding window over a `5×5` feature map is rearranged into a matrix for fast GEMM.

## Probability and Statistics

> **Planned content:** Probability distributions: Gaussian, Categorical, Bernoulli, Beta. Maximum Likelihood Estimation (MLE). Bayes' theorem. KL divergence: `D_KL(P||Q) = Σ P log(P/Q)`. Jensen-Shannon divergence. Entropy. Cross-entropy as negative log-likelihood. ELBO (Evidence Lower Bound) for VAEs. Monte Carlo estimation.

> **📊 Planned diagram:** KL divergence visualization — two distributions P and Q, showing where KL measures their difference.

## Calculus and Optimization

> **Planned content:** Chain rule: the backbone of backpropagation. Gradient descent: `θ ← θ - η ∇_θ L`. Stochastic gradient descent. Momentum. Adam/AdamW. Second-order methods (briefly). The Jacobian: `∂output/∂input` for a vector-valued function. Hessian and its role in optimization landscape. Why flat minima generalize better.

> **📊 Planned diagram:** Backpropagation through a convolutional layer — showing how gradients flow from loss → output → kernel weights.

## Geometry for CV

> **Planned content:** Homogeneous coordinates. Projection matrix. Rotation matrices and SO(3). Quaternions for rotation. Rigid body transforms (rotation + translation). Fundamental and Essential matrices (stereo). Epipolar geometry. Camera calibration (intrinsics + extrinsics). Homography estimation.

> **📊 Planned diagram:** Camera coordinate systems — world frame → camera frame → image plane → pixel coordinates, with transformation matrices at each step.

## Fourier Transform for CV

> **Planned content:** Discrete Fourier Transform (DFT). Frequency domain: low (coarse structure) vs. high (edges, texture) frequencies. 2D DFT of an image. Convolution theorem: `conv(f, g) = IDFT(DFT(f) · DFT(g))`. Applications: fast convolution, frequency-based analysis, NeRF positional encoding.

> **📊 Planned diagram:** Image and its 2D DFT magnitude spectrum — showing center = DC (low frequency), edges = high frequency. Filtered reconstructions.

## Information Theory Basics

> **Planned content:** Entropy: `H(X) = -Σ p log p`. Cross-entropy: `H(P, Q) = -Σ P log Q`. Mutual information. Why cross-entropy = NLL of categorical distribution. The AUC-ROC area and information theory connection.

**[← Back to Table of Contents](./README.md)**

---
*Last updated: May 2026*
