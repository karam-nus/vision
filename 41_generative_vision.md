---
title: "Chapter 41 — Generative Vision"
---

[← Back to Table of Contents](./README.md)

# Chapter 41 — Generative Vision

> *"Understanding vision at its deepest means being able to generate it. Generative models close the loop between perception and creation."*

## The Generative Vision Landscape

> **Planned content:** Three families: GANs (adversarial), VAEs (variational), Diffusion (score-based). Each models the data distribution differently. Recent convergence around diffusion models. Applications: text-to-image, image editing, super-resolution, inpainting, video generation, 3D generation, synthetic training data.

> **📊 Planned diagram:** Generative model taxonomy — GAN (discriminator-generator adversarial) vs. VAE (encoder-decoder probabilistic) vs. Diffusion (iterative denoising) — conceptual comparison.

## GANs: Generative Adversarial Networks

> **Planned content:** Minimax game: generator G creates fake images, discriminator D distinguishes real/fake. Objective: `min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]`. Training instability (mode collapse, vanishing gradients). DCGAN: convolutional architecture. Progressive GAN: growing resolution during training. StyleGAN2/3: style-based generator, high-quality portraits. BigGAN: class-conditional at ImageNet scale.

> **📊 Planned diagram:** GAN training loop — noise z → G → fake image → D competes with real images → gradient back to both G and D.

> **📊 Planned diagram:** StyleGAN2 architecture — mapping network (z → w) + synthesis network (constant input → styled convolutions → image), with AdaIN/modulation.

## VAEs: Variational Autoencoders

> **Planned content:** Encoder: `q(z|x)` → μ, σ. Reparameterization trick: `z = μ + σ * ε`. Decoder: `p(x|z)`. ELBO loss: reconstruction + KL divergence. VQ-VAE: discrete latent codes via vector quantization. VQ-VAE-2: hierarchical latent codes for high resolution. Used as perceptual compression in latent diffusion.

> **📊 Planned diagram:** VAE architecture — encoder `[B, 3, H, W]` → μ `[B, D]` + log σ `[B, D]` → sample z `[B, D]` → decoder → reconstruction `[B, 3, H, W]`. KL loss on latent distribution.

## DDPM: Denoising Diffusion Probabilistic Models

> **Planned content:** Forward process: gradually add Gaussian noise over T=1000 steps. `q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)`. Closed form: `q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)`. Reverse process: denoise step by step. Network ε_θ predicts the noise at each step. Training: MSE on predicted vs. actual noise. Sampling: T denoising steps. Connection to score matching.

> **📊 Planned diagram:** DDPM forward and reverse process — x_0 (clean) → x_1 → ... → x_T (pure noise) forward, and reverse denoising x_T → ... → x_0 with U-Net noise predictor.

$$\mathcal{L}_{DDPM} = E_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)\|^2\right]$$

## DDIM: Faster Sampling

> **Planned content:** DDIM: deterministic non-Markovian reverse process. Skip T → 50 steps while maintaining quality. PLMS, DPM-Solver: even faster samplers. The quality-speed tradeoff in sampling steps. CFG (Classifier-Free Guidance): `ε_guided = ε_uncond + w * (ε_cond - ε_uncond)`. Guidance scale w.

> **📊 Planned diagram:** CFG mechanism — conditional (text-guided) noise prediction pushed further from unconditional prediction, amplifying text alignment.

## Stable Diffusion: Latent Diffusion Models

> **Planned content:** Key insight: diffusion in latent space (4× smaller) instead of pixel space. VQ-VAE encoder: `[B, 3, 512, 512]` → `[B, 4, 64, 64]` latent. U-Net diffusion in latent space. CLIP text encoder for text conditioning via cross-attention. Decoder: `[B, 4, 64, 64]` → `[B, 3, 512, 512]`. SDXL: two text encoders (CLIP-L + OpenCLIP-G), larger U-Net. SD3: DiT-based (transformer backbone).

> **📊 Planned diagram:** Stable Diffusion architecture — CLIP text encoder → text embeddings → U-Net (with cross-attention on text) operating on latent `[B, 4, 64, 64]` → VAE decoder → image.

## ControlNet: Conditioned Generation

> **Planned content:** Add spatial conditioning (pose, depth, canny, segmentation map, normal map) to Stable Diffusion. ControlNet architecture: copies SD encoder weights, connects via zero-conv layers. Training only the ControlNet copy. Enables precise control over generated image structure.

> **📊 Planned diagram:** ControlNet — condition image (pose/depth/edge) + text prompt → modified U-Net with ControlNet branch → conditioned generation.

## Video Generation

> **Planned content:** Make-A-Video, Imagen Video, CogVideoX. Sora: diffusion transformer (DiT) for video. Challenges: temporal consistency, long duration. Factored space-time attention. Training data: long video + text captions.

## Evaluating Generative Models

> **Planned content:** FID: lower = better, measures distributional distance. IS: higher = better (quality + diversity). Precision/Recall/Density/Coverage. LPIPS: perceptual similarity. CLIP-Score for text-image alignment. Human preference studies (ELO, A/B tests).

**Next: [Chapter 42 — The Frontier →](./42_frontier.md)**

---
*Last updated: May 2026*
