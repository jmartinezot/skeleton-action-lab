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
   
---

## 🐳 Build and Run the Docker Environment

The repository includes a ready-to-use Dockerfile  
[`Dockerfile.msg3d-ctrgcn-apex`](Dockerfile.msg3d-ctrgcn-apex)
that installs **MS-G3D**, **CTR-GCN**, and **NVIDIA Apex** in one image.

### 🧱 Build the image

From the root of this repository:

```bash
docker build -t skeleton-lab:latest -f Dockerfile.msg3d-ctrgcn .
```

This step downloads the PyTorch 2.3 CUDA 12.1 runtime image,
and clones both model repositories.

### 🚀 Run the container (CTR-GCN with .npz)

Mount your dataset and work directory from the host:

```bash
docker run -it --rm --gpus all \
  -v /home/user/Datasets/NTU60_npz:/workspace/CTR-GCN/data/ntu \
  -v /home/user/MS-G3D_workdir:/workspace/work_dir \
  skeleton-lab:latest
``

Once inside the container:

```bash
cd /workspace/CTR-GCN
```

# Example: train CTR-GCN on NTU60 cross-subject split

```bash
python main.py \
  --config ./config/nturgbd-cross-subject/train_joint.yaml \
  --work-dir ../work_dir/ctrgcn_ntu60_xsub_joint
```

### 🧪 Run the container (MS-G3D with raw .skeleton files)

If you later download the original NTU RGB+D 60 skeletons:

```bash
docker run -it --rm --gpus all \
  -v /home/user/Datasets/NTU_RGBD60/nturgb+d_skeletons:/workspace/data/nturgbd_raw/nturgb+d_skeletons \
  -v /home/user/Datasets/NTU_RGBD60/NTU_RGBD_samples_with_missing_skeletons.txt:/workspace/data/nturgbd_raw/NTU_RGBD_samples_with_missing_skeletons.txt:ro \
  -v /home/user/Datasets/NTU_RGBD60_processed:/workspace/MS-G3D/data \
  -v /home/user/MS-G3D_workdir:/workspace/work_dir \
  skeleton-lab:latest
```

Inside:

```bash
cd /workspace/MS-G3D
cd data_gen && python ntu_gendata.py && cd ..
python main.py \
  --config config/nturgbd-cross-subject/train_joint.yaml \
  --work-dir work_dir/ntu60/xsub/msg3d_joint \
  --device 0 \
  --half
```

### 🧩 Notes

- --gpus all enables CUDA inside the container (requires nvidia-container-toolkit).

- Adjust all /home/user/... paths to match your own directories.

- All experiment outputs are saved under your mounted work_dir folder.

