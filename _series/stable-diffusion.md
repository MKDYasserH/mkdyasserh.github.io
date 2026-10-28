---
title: Stable Diffusion from Scratch
tagline: Build a text-to-image model one block at a time.
description: Convolutions, attention, VAEs, diffusion, UNet and CLIP, explained with the maths and implemented in TensorFlow. Every part builds on the one before it.
cover: assets/img/stable_diffusion.png
level: Intermediate
status: complete
order: 1
---

Stable Diffusion looks like magic until you take it apart. This series does exactly that: we start from the building blocks every modern generative model reuses (convolutions, attention, autoencoders), then assemble them into the full latent diffusion pipeline with classifier-free guidance, a UNet denoiser and CLIP text conditioning.

Each part mixes intuition, the equations that matter, and code you can run. Read them in order if you're new to generative models, or jump straight to the block you need.
