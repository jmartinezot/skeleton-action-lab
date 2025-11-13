# Skeleton Action Lab 🧠

> ⚠️ **Heads-up:** this repository is under active, heavy development. APIs, scripts, and
> training recipes are likely to change without notice and may occasionally break.

This repository provides a **Dockerized research environment** for experimenting with
**skeleton-based action recognition** on the NTU RGB+D 60 dataset. Each model ships with its
own Dockerfile so you can build lean, model-specific images. For example, `ctrgcn.docker`
builds a container with the CTR-GCN dependencies and `msg3d.docker` provisions the
MS-G3D toolchain, while additional Dockerfiles can be added alongside them for other
backbones.

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
  - `Dockerfile.msg3d-ctrgcn` – optional legacy combo image kept for cross-checking multi-model workflows
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

## 📦 Data Preparation

The experiments expect the NTU RGB+D 60 dataset to live outside the container in a host
directory named `Datasets`. Within that folder, organize the raw download and all
preprocessed variants in an `NTU60` directory so each model can reuse the same source
files:

```
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
        └── NTU60_CS.npz   (link to kaggle raw)
```

> **Tip:** the `ctrgcn/NTU60_CS.npz` entry can simply be a symbolic link back to the
> `kaggle_raw/NTU60_CS.npz` file so you avoid storing duplicates while keeping each model's
> expected file layout intact.

Helpful scripts for inspecting and converting the dataset:

- `dataset_tools/inspect_npz.py` – dump keys, shapes, and dtypes in an `.npz` file.
- `convert_ntu60_kaggle_to_ctrgcn.py` – convert Kaggle layout to CTR-GCN layout.
- `convert_ctr_npz_to_msg3d.py` – bridge from CTR-GCN `.npz` tensors to the MS-G3D
  preprocessing pipeline.

When launching containers, bind-mount `~/Datasets/NTU60` into the expected path inside
the container (see the `docker run` examples below). Keeping all derived formats together
makes it easy to reuse conversions across experiments.

---

## 🚀 Quick Start (CTR-GCN with `.npz`)

1. Place your `NTU60_CS.npz` in a folder on the host, e.g.:

   ```bash
   mkdir -p /home/bob/Datasets/NTU60/kaggle_raw
   cp NTU60_CS.npz /home/bob/Datasets/NTU60/kaggle_raw/
   
---

## 🐳 Build and Run the Docker Environments

Pick the Dockerfile that matches the model you want to explore. Each Dockerfile lives at the
repository root and builds a self-contained environment for that model. For example,
[`ctrgcn.docker`](ctrgcn.docker) installs only the CTR-GCN stack and
[`msg3d.docker`](msg3d.docker) focuses solely on MS-G3D, while
[`Dockerfile.msg3d-ctrgcn`](Dockerfile.msg3d-ctrgcn) remains available if you still need the
combined setup.

### 🧱 Build the image

From the root of this repository:

```bash
# Build the CTR-GCN-only image
docker build -t skeleton-lab:ctrgcn -f ctrgcn.docker .

# Build the MS-G3D-only image
docker build -t skeleton-lab:msg3d -f msg3d.docker .

# (Optional) build the combined MS-G3D + CTR-GCN image for comparison
docker build -t skeleton-lab:msg3d-ctrgcn -f Dockerfile.msg3d-ctrgcn .
```

Add more build commands as you introduce Dockerfiles for new models. Each build downloads the
base PyTorch 2.3 CUDA 12.1 runtime image and installs the dependencies required by that
specific model.

### 📜 Helper scripts included in the image

Several utility scripts ship with the Docker image to make quick GPU and dataset
checks easier. They live at the repository root so the `COPY` instruction in the
Dockerfile can add them to `/workspace/` inside the container:

| Script | Purpose |
|--------|---------|
| `check_gpu.py` | Prints CUDA availability, device count, device name, and the PyTorch/CUDA versions so you can confirm the container sees your GPU. |
| `dataset_tools/inspect_npz.py` | Dumps the keys, shapes, and dtypes in an NTU60 `.npz` archive (defaults to `/workspace/CTR-GCN/data/ntu/NTU60_CS.npz`) to verify downloads and mounts. |
| `convert_ntu60_kaggle_to_ctrgcn.py` | Converts the Kaggle-style `NTU60_CS.npz` (`(N, T, D=150)` with one-hot labels) into the `(N, C, T, V, M)` layout used by CTR-GCN and saves `NTU60_CS_ctrgcn.npz`. |
| `skeleton_dataset_ctrgcn.py` | Defines a PyTorch `Dataset` for the converted skeleton tensors, optionally dropping the second person stream, and includes a loader sanity check. |
| `train_npz_mlp.py` | Runs a lightweight MLP baseline directly on the Kaggle `.npz` file to stress-test your GPU/VRAM and confirm the data loads. |
| `train_skeleton_gcn.py` | Trains a toy spatio-temporal baseline built on `skeleton_dataset_ctrgcn.py` to validate the end-to-end skeleton pipeline. |

Feel free to run any of these scripts inside the container (e.g., `python check_gpu.py`).

### 🚀 Run the container (CTR-GCN with .npz)

Mount your dataset and work directory from the host:

```bash
docker run -it --rm --gpus all \
  --shm-size=8g \
  --mount type=bind,source="$HOME/Datasets/NTU60/ctrgcn",target=/workspace/CTR-GCN/data/ntu \
  skeleton-lab:ctrgcn
```

Once inside the container:

```bash
cd /workspace/CTR-GCN
```

## Example: train CTR-GCN on NTU60 cross-subject split

```bash
python main.py \
  --config ./config/nturgbd-cross-subject/default.yaml \
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
  skeleton-lab:msg3d
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

---

## 🔭 Future Work

- Reintroduce **NVIDIA Apex** for mixed-precision training once the training stacks stabilize.
- Expand the data tooling to cover more skeleton benchmarks beyond NTU RGB+D 60.

- Adjust all /home/user/... paths to match your own directories.

- All experiment outputs are saved under your mounted work_dir folder.

