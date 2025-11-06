# MS-G3D + Apex Research Environment 🧠

This repository provides a **ready-to-run Dockerized setup** for experimenting with
**skeleton-based action recognition** using the [MS-G3D](https://github.com/kenziyuliu/MS-G3D)
model (Multi-Scale Graph 3D Network, CVPR 2020) with **NVIDIA Apex** for
mixed-precision training on GPUs.

It is designed as a **research-friendly playground** for:

- 🧍‍♂️ **Action classification** from 3D pose sequences  
- ⏩ **Action anticipation** / next-action prediction  
- 💬 Extensions to **multimodal or question-answering** pipelines (e.g., combining skeletons with image descriptors or text)

---

## ✨ Features

- 🐳 **Docker-based** setup for clean, reproducible experiments  
- ⚙️ **Preinstalled NVIDIA Apex** for memory-efficient mixed-precision training (`--half` flag)  
- 📊 **MS-G3D** implementation ready for the NTU RGB+D 60 dataset  
- 🧩 Easily extendable for new tasks (anticipation, multimodal fusion, self-supervised learning)

---

## 🧱 Stack Overview

| Component | Purpose |
|------------|----------|
| **PyTorch 2.3 + CUDA 12.1** | Deep learning framework |
| **MS-G3D** | Backbone model for skeleton-based action recognition |
| **NVIDIA Apex** | Mixed-precision and fused-kernel training |
| **NTU RGB+D 60** | Public benchmark dataset for 3D human actions |
| **Docker** | Reproducible containerized environment |

---

## 🚀 Quick Start

```bash
# Build the image
docker build -t msg3d-apex:latest -f Dockerfile.msg3d-apex .

# Run container with GPU
docker run -it --rm --gpus all \
  -v /path/to/NTU_RGBD60:/workspace/data/nturgbd_raw/nturgb+d_skeletons \
  msg3d-apex:latest

