# Skeleton Action Lab 🧠

> ⚠️ **Heads-up:** this repository is under active, heavy development. APIs, scripts, and
> training recipes may change without notice and occasionally break.

This repository provides a **Dockerized research environment** for experimenting with
**skeleton-based action recognition** on the NTU RGB+D 60 dataset. Each model ships with its
own Dockerfile so you can build lean, model-specific images (e.g., `ctrgcn.docker` for
CTR-GCN, `msg3d.docker` for MS-G3D) while leaving room to add more backbones.

## 🧭 Table of Contents

- [Features](#-features)
- [Stack Overview](#-stack-overview)
- [Prerequisites](#-prerequisites)
- [Data Preparation](#-data-preparation)
- [Build & Run: CTR-GCN](#-build--run-ctr-gcn)
- [Build & Run: MS-G3D](#-build--run-ms-g3d)
- [Helper Scripts & Benchmarks](#-helper-scripts--benchmarks)
- [Future Work](#-future-work)

Currently included models:

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
- 🧠 **Model-specific Dockerfiles** (build only what you need):
  - `ctrgcn.docker` – builds an image tailored for CTR-GCN experiments with preprocessed `.npz` tensors
  - `msg3d.docker` – installs the MS-G3D pipeline (data generation + training) without CTR-GCN extras
  - Add new Dockerfiles (e.g., `stgcn.docker`, `mim.docker`) as more models are integrated
- 🧱 Built on **PyTorch 2.3 + CUDA 12.1** runtime

---

## 🧱 Stack Overview

| Component       | Purpose                                           |
|----------------|---------------------------------------------------|
| **PyTorch**    | Deep learning framework                           |
| **MS-G3D**     | Multi-scale ST-GCN for skeleton action recognition |
| **CTR-GCN**    | GCN with channel-wise topology refinement         |
| **NTU RGB+D 60** | Benchmark dataset for 3D human actions          |
| **Docker**     | Containerized environment                         |

---

## 🧰 Prerequisites

- **Hardware**: NVIDIA GPU with CUDA support (tested around ~8 GB VRAM)
- **Drivers/Runtime**: NVIDIA drivers + `nvidia-container-toolkit` configured for Docker
- **Tools**: Docker (Linux containers), Bash/Python 3
- **Host paths** (customize as needed):
  - `$HOME/Datasets/NTU60/` – root folder for all NTU RGB+D 60 assets
  - `$HOME/Datasets/NTU60/kaggle_raw/NTU60_CS.npz` – Kaggle download
  - `$HOME/Datasets/NTU60/ctrgcn/NTU60_CS.npz` – CTR-GCN layout (can be a symlink to Kaggle raw)
  - `$HOME/Datasets/NTU60/msg3d/` – MS-G3D layout outputs
  - `$HOME/MS-G3D_workdir` – writable work directory for MS-G3D runs

## 📦 Data Preparation

The experiments expect the NTU RGB+D 60 dataset to live outside the container in
`$HOME/Datasets/NTU60`. Keep raw downloads and converted formats together so each model can
reuse the same source files.

```text
Datasets/
└── NTU60/
    ├── kaggle_raw/
    │   └── NTU60_CS.npz
    ├── skeleton5d/
    │   └── NTU60_CS_skeleton5d.npz
    ├── msg3d/
    │   └── xsub/
    │       ├── train_data_joint.npy
    │       ├── train_label.pkl
    │       ├── val_data_joint.npy
    │       └── val_label.pkl
    └── ctrgcn/
        └── NTU60_CS.npz   (symlink to Kaggle raw is fine)
```

### Kaggle download

- Download `NTU60_CS.npz` from Kaggle: https://www.kaggle.com/datasets/jarex616/ntu-rgb-d-60-skeleton-data-npz
- Place it at `$HOME/Datasets/NTU60/kaggle_raw/NTU60_CS.npz`.

```bash
mkdir -p "$HOME/Datasets/NTU60/kaggle_raw"
cp NTU60_CS.npz "$HOME/Datasets/NTU60/kaggle_raw/"
```

### Convert to CTR-GCN layout

Convert the Kaggle archive to the `(N, C, T, V, M)` format expected by CTR-GCN (CPU-friendly):

```bash
python3 convert_ntu60_kaggle_to_ctrgcn.py \
  --input "$HOME/Datasets/NTU60/kaggle_raw/NTU60_CS.npz" \
  --output "$HOME/Datasets/NTU60/ctrgcn/NTU60_CS.npz"
```

### Convert to skeleton5d layout (for `train_skeleton_gcn.py`)

Some quick-start scripts, such as `train_skeleton_gcn.py`, expect a 5D tensor layout.
You can generate this format inside the CTR-GCN container using the provided conversion
utility:

```bash
docker run -it --rm --gpus all \
  --shm-size=8g \
  --mount type=bind,source="$HOME/Datasets/NTU60/",target=/workspace/CTR-GCN/data/ntu \
  skeleton-lab:ctrgcn

python3 /workspace/scripts/dataset_tools/convert_kaggle_to_skeleton5d.py \
  --input /workspace/CTR-GCN/data/ntu/ctrgcn/NTU60_CS.npz \
  --output /workspace/CTR-GCN/data/ntu/skeleton5d/NTU60_CS_skeleton5d.npz
```

You can test it:

```bash
python3 /workspace/scripts/train_skeleton_gcn.py \
  --npz-path /workspace/CTR-GCN/data/ntu/skeleton5d/NTU60_CS_skeleton5d.npz \
  --epochs 5 --batch-size 32 --device cuda:0
```

### Convert CTR-GCN layout to MS-G3D layout

Generate the `xsub` splits for MS-G3D from the CTR-GCN tensor (runs inside or outside a
container):

```bash
python3 convert_ctr_npz_to_msg3d.py \
  --input "$HOME/Datasets/NTU60/ctrgcn/NTU60_CS.npz" \
  --output-root "$HOME/Datasets/NTU60/msg3d"
```

Bind-mount `$HOME/Datasets/NTU60` into the container paths shown below so both models see
their expected layouts.

---
# 🚀 Build & Run: CTR-GCN

CTR-GCN uses preprocessed `.npz` tensors. Use the mount pattern below so `/workspace/CTR-GCN/data/ntu/NTU60_CS.npz` resolves inside the container.

### Build image

```bash
docker build -t skeleton-lab:ctrgcn -f ctrgcn.docker .
```

### Run container

```bash
docker run -it --rm --gpus all \
  --shm-size=8g \
  --mount type=bind,source="$HOME/Datasets/NTU60/ctrgcn",target=/workspace/CTR-GCN/data/ntu \
  skeleton-lab:ctrgcn
```

Inside the container:

```bash
cd /workspace/CTR-GCN
```

### Example training commands

Train CTR-GCN on the NTU60 cross-subject split:

```bash
python3 main.py \
  --config ./config/nturgbd-cross-subject/default.yaml \
  --work-dir ../work_dir/ctrgcn_ntu60_xsub_joint
```

Run lightweight sanity checks (inside the same container):

```bash
# GPU sanity check
python3 /workspace/scripts/check_gpu.py

# Inspect Kaggle npz contents
python3 /workspace/scripts/dataset_tools/inspect_npz.py /workspace/CTR-GCN/data/ntu/NTU60_CS.npz

# Toy baselines
python3 /workspace/scripts/train_npz_mlp.py --npz-path /workspace/CTR-GCN/data/ntu/NTU60_CS.npz --epochs 5 --batch-size 256 --device cuda:0
```

# 🚀 Build & Run: MS-G3D

MS-G3D can consume the converted CTR-GCN tensor or the original `.skeleton` files. The
example below assumes the converted `.npz` workflow with a work directory mounted at
`/workspace/work_dir`.

### Build image

```bash
docker build -t skeleton-lab:msg3d -f msg3d.docker .
```

### Run container

```bash
docker run -it --rm --gpus all \
  --mount type=bind,source="$HOME/Datasets/NTU60/msg3d",target=/workspace/MS-G3D/data/ntu \
  skeleton-lab:msg3d
```

If using raw `.skeleton` files instead, bind the raw dataset and processed output paths in
the same pattern as above (replace mounts with your paths).

### Example training commands

Convert CTR-GCN `.npz` to the MS-G3D `xsub` layout (writes to `/workspace/MS-G3D/data/ntu/xsub/`):

```bash
cd /workspace
python3 convert_ctr_npz_to_msg3d.py
```

Train MS-G3D on the cross-subject split:

```bash
cd /workspace/MS-G3D
python3 main.py \
  --config config/nturgbd-cross-subject/train_joint.yaml \
  --work-dir /workspace/work_dir/ntu60_xsub_msg3d_joint \
  --device 0 \
  --half
```

To use raw `.skeleton` files, run the built-in generator and launch training (same mount
pattern as above, just pointing to raw data):

```bash
cd /workspace/MS-G3D
cd data_gen && python3 ntu_gendata.py && cd ..
python3 main.py \
  --config config/nturgbd-cross-subject/train_joint.yaml \
  --work-dir work_dir/ntu60/xsub/msg3d_joint \
  --device 0 \
  --half
```

---

## 🛠 Helper Scripts & Benchmarks

All Python utilities at the repository root are copied into `/workspace/scripts` during the
Docker build for quick access inside containers.

### Dataset tools and sanity checks

| Script | Purpose |
|--------|---------|
| `check_gpu.py` | Print CUDA availability, device count, device name, and PyTorch/CUDA versions. |
| `dataset_tools/inspect_npz.py` | Dump keys, shapes, and dtypes in an NTU60 `.npz` archive (defaults to `/workspace/CTR-GCN/data/ntu/NTU60_CS.npz`). |
| `convert_ntu60_kaggle_to_ctrgcn.py` | Convert the Kaggle-style `(N, T, D=150)` file into the `(N, C, T, V, M)` layout used by CTR-GCN. |
| `convert_ctr_npz_to_msg3d.py` | Bridge from the CTR-GCN tensor to the MS-G3D `xsub` preprocessing pipeline. |

### Baselines and experiments

| Script | Purpose |
|--------|---------|
| `skeleton_dataset_ctrgcn.py` | PyTorch `Dataset` for converted skeleton tensors with loader sanity checks. |
| `train_npz_mlp.py` | Lightweight MLP baseline on the Kaggle `.npz` file (good VRAM/profiling sanity check). |
| `train_skeleton_gcn.py` | Toy spatio-temporal baseline built on `skeleton_dataset_ctrgcn.py`. |
| `train_skeleton_gcn_simple_backbone.py` | Simplified backbone variant for quick iterations. |
| `train_ctrgcn_npz.py` / `train_ctrgcn_npz_full.py` | CTR-GCN training entry points for `.npz` data. |
| `train_ctrgcn_npz_with_dloader.py` | CTR-GCN training with a data loader abstraction. |

### Benchmarks

- `benchmark_models.py` – synthetic training loops to confirm imports/device setup and measure wall-clock timing.
  - Run on CPU or GPU with `--device cpu`, `--device cuda`, or `--device cuda:0`.
  - `--steps` controls the number of training iterations; `--seq_len` controls the synthetic sequence length.
  - Example: `python3 benchmark_models.py --model ctrgcn --steps 50 --seq_len 100 --device cuda:0 --amp`.

---

## 🔭 Future Work

- Reintroduce **NVIDIA Apex** for mixed-precision training once the stacks stabilize.
- Expand data tooling to more skeleton benchmarks beyond NTU RGB+D 60.
- Adjust host paths to match your environment; all experiment outputs land in your mounted `work_dir` folders.

