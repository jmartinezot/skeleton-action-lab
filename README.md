# Skeleton Action Lab 🧠

This repository provides a **Dockerized research environment** for experimenting with
**skeleton-based action recognition** on the NTU RGB+D 60 dataset using two state-of-the-art
graph-based models:

- **MS-G3D** (Multi-Scale Graph 3D Network, CVPR 2020) – classic multi-scale ST-GCN backbone  
- **CTR-GCN** (Channel-wise Topology Refinement Graph Convolutional Network, ICCV 2021) – strong, widely used baseline that operates directly on preprocessed `.npz` files

The environment is designed to answer a very practical question:

> *“Can my machine (single GPU, ~8 GB VRAM) run modern skeleton models comfortably?”*

It is also a starting point for research on:

- 🧍‍♂️ **Action classification** from pose sequences  
- ⏩ **Action anticipation** / next-action prediction (via modified sampling or heads)  
- 🧩 **Multimodal extensions** (e.g. adding image descriptors to skeletons)

---

## ✨ Features

- 🐳 **Docker-based**: reproducible experiments in a single container  
- ⚙️ **NVIDIA Apex** preinstalled for mixed-precision training (saves VRAM, speeds up training)  
- 🧠 **Two backbones in one image**:
  - `MS-G3D/` – expects original NTU `.skeleton` files (raw skeleton format)  
  - `CTR-GCN/` – works directly with `NTU60_CS.npz` / `NTU60_CV.npz` (preprocessed format)  
- 🧱 Built on **PyTorch 2.3 + CUDA 12.1** runtime

---

## 🧱 Stack Overview

| Component       | Purpose                                           |
|----------------|---------------------------------------------------|
| **PyTorch**    | Deep learning framework                           |
| **MS-G3D**     | Multi-scale ST-GCN for skeleton action recognition |
| **CTR-GCN**    | GCN with channel-wise topology refinement         |
| **NVIDIA Apex**| Mixed precision + fused ops for efficient training|
| **NTU RGB+D 60** | Benchmark dataset for 3D human actions          |
| **Docker**     | Containerized environment                         |

---

## 🚀 Quick Start (CTR-GCN with `.npz`)

1. Place your `NTU60_CS.npz` in a folder on the host, e.g.:

   ```bash
   mkdir -p /home/bob/Datasets/NTU60_npz
   cp NTU60_CS.npz /home/bob/Datasets/NTU60_npz/

