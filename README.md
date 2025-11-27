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
- [Build & Run: Baselines](#-build--run-baselines)
- [Build & Run: MS-G3D](#-build--run-ms-g3d)
- [Build & Run: FreqMixFormer](#-build--run-freqmixformer)
- [Build & Run: SkateFormer](#-build--run-skateformer)
- [Helper Scripts & Benchmarks](#-helper-scripts--benchmarks)
- [Future Work](#-future-work)

Currently included models:

- **MS-G3D** (Multi-Scale Graph 3D Network, CVPR 2020) – classic multi-scale ST-GCN backbone
- **CTR-GCN** (Channel-wise Topology Refinement Graph Convolutional Network, ICCV 2021) – strong, widely used baseline that operates directly on preprocessed `.npz` files
- **FreqMixFormer** (Frequency Guidance Matters: Skeletal Action Recognition by
Frequency-Aware Mixed Transformer, ACM MM 2024) – frequency-aware mixed transformer that now reads the same NTU60 tensors (`kaggle_raw` one-hot or converted CTR-GCN 5D)
- **SkateFormer** (SkateFormer: Skeletal-Temporal Transformer for Human Action Recognition, ECCV 2024) – skeletal-temporal transformer that trains directly on the Kaggle-format NTU60 tensor (`NTU60_CS.npz`)

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
  - `ctrgcn.docker` – builds an image tailored for CTR-GCN experiments using the Kaggle `.npz` directly
  - `msg3d.docker` – installs the MS-G3D pipeline (data generation + training) without CTR-GCN extras
  - `freqmixformer.docker` – ships the FreqMixFormer codebase configured to read the shared NTU60 tensors
  - `skateformer.docker` – bundles the SkateFormer transformer that trains directly on the Kaggle NPZ
  - `baselines.docker` – lightweight sanity checks and toy baselines (MLP, ST-GCN-lite, synthetic benchmarks)
  - Add new Dockerfiles (e.g., `stgcn.docker`, `mim.docker`) as more models are integrated
- 🧱 Built on **PyTorch 2.3 + CUDA 12.1** runtime

---

## 🧱 Stack Overview

| Component       | Purpose                                           |
|----------------|---------------------------------------------------|
| **PyTorch**    | Deep learning framework                           |
| **MS-G3D**     | Multi-scale ST-GCN for skeleton action recognition |
| **CTR-GCN**    | GCN with channel-wise topology refinement         |
| **FreqMixFormer** | Frequency-aware mixed transformer for skeleton actions |
| **SkateFormer** | Skeletal-temporal transformer that reads the Kaggle NPZ directly |
| **NTU RGB+D 60** | Benchmark dataset for 3D human actions          |
| **Docker**     | Containerized environment                         |

---

## 🧰 Prerequisites

- **Hardware**: NVIDIA GPU with CUDA support (tested around ~8 GB VRAM)
- **Drivers/Runtime**: NVIDIA drivers + `nvidia-container-toolkit` configured for Docker
- **Tools**: Docker (Linux containers), Bash/Python 3
- **Host paths** (customize as needed):
  - `$HOME/Datasets/NTU60/` – root folder for all NTU RGB+D 60 assets
  - `$HOME/Datasets/NTU60/kaggle_raw/NTU60_CS.npz` – Kaggle download (one file drives all stacks here)
  - `$HOME/Datasets/NTU60/ctrgcn/NTU60_CS.npz` – optional symlink/copy to the Kaggle file for convenience
  - FreqMixFormer and CTR-GCN both accept the Kaggle NPZ directly; mount `$HOME/Datasets/NTU60` into `/workspace/data/NTU60` for FreqMixFormer or bind the single file for CTR-GCN
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
    └── ctrgcn/
        └── NTU60_CS.npz   (symlink/copy to Kaggle raw; optional)
```

### What each format contains

- Kaggle raw (`Datasets/NTU60/kaggle_raw/NTU60_CS.npz`): keys `x_train`, `y_train`, `x_test`, `y_test`. Shapes `x_* = (N, T, D=150)` where `150 = 25 joints × 3 coords (x,y,z) × up to 2 persons` flattened; `y_*` are one-hot class labels (60 classes, cross-subject split). Only 3D joints—no orientations or camera metadata.
- CTR-GCN loader accepts the Kaggle NPZ directly. If you keep a separate path, symlink/copy the Kaggle file to `Datasets/NTU60/ctrgcn/NTU60_CS.npz`.
- Skeleton5D quickstart (`Datasets/NTU60/skeleton5d/NTU60_CS_skeleton5d.npz`): the same `(N, 3, T, 25, 2)` tensor and labels packaged for toy scripts like `train_skeleton_gcn.py`. Not used by MS-G3D.

All current workflows use only 3D joint coordinates (and labels); hand states, quaternions, and camera metadata from raw `.skeleton` files are not used here.

> FreqMixFormer can read either the Kaggle one-hot file (`/workspace/data/NTU60/kaggle_raw/NTU60_CS.npz`) or a CTR-GCN-path symlink of that file. Point the config to whichever path you mount.

### Kaggle download

- Download `NTU60_CS.npz` from Kaggle: https://www.kaggle.com/datasets/jarex616/ntu-rgb-d-60-skeleton-data-npz
- Place it at `$HOME/Datasets/NTU60/kaggle_raw/NTU60_CS.npz`.

```bash
mkdir -p "$HOME/Datasets/NTU60/kaggle_raw"
cp NTU60_CS.npz "$HOME/Datasets/NTU60/kaggle_raw/"
```

### (Optional) convert to CTR-GCN layout

CTR-GCN and most loaders here can read the Kaggle NPZ directly. If you still want a
separate CTR-GCN path, you can convert or simply symlink/copy:

```bash
mkdir -p "$HOME/Datasets/NTU60/ctrgcn"
ln -s "$HOME/Datasets/NTU60/kaggle_raw/NTU60_CS.npz" "$HOME/Datasets/NTU60/ctrgcn/NTU60_CS.npz"
# or use the converter to rewrite the layout if desired:
# python3 convert_ntu60_kaggle_to_ctrgcn.py --input "$HOME/Datasets/NTU60/kaggle_raw/NTU60_CS.npz" --output "$HOME/Datasets/NTU60/ctrgcn/NTU60_CS.npz"
```

### Convert to skeleton5d layout (for `train_skeleton_gcn.py`)

Some quick-start scripts, such as `train_skeleton_gcn.py`, expect a 5D tensor layout.
You can generate this format inside the CTR-GCN container using the provided conversion
utility:

```bash
docker run -it --rm --gpus all \
  --shm-size=8g \
  --mount type=bind,source="$HOME/Datasets/NTU60/kaggle_raw/NTU60_CS.npz",target=/workspace/CTR-GCN/data/ntu/NTU60_CS.npz,readonly \
  skeleton-lab:ctrgcn

python3 /workspace/scripts/dataset_tools/convert_kaggle_to_skeleton5d.py \
  --input /workspace/CTR-GCN/data/ntu/NTU60_CS.npz \
  --output /workspace/CTR-GCN/data/ntu/skeleton5d/NTU60_CS_skeleton5d.npz
```

You can test it:

```bash
python3 /workspace/scripts/train_skeleton_gcn.py \
  --npz-path /workspace/CTR-GCN/data/ntu/skeleton5d/NTU60_CS_skeleton5d.npz \
  --epochs 5 --batch-size 32 --device cuda:0
```

---
# 🚀 Build & Run: CTR-GCN

CTR-GCN can read the Kaggle NTU60 NPZ directly (no extra conversion required). Bind-mount
the Kaggle file into `/workspace/CTR-GCN/data/ntu/NTU60_CS.npz`.

### Build image

```bash
docker build -t skeleton-lab:ctrgcn -f ctrgcn.docker .
```

### Run container

```bash
docker run -it --rm --gpus all \
  --shm-size=8g \
  --mount type=bind,source="$HOME/Datasets/NTU60/kaggle_raw/NTU60_CS.npz",target=/workspace/CTR-GCN/data/ntu/NTU60_CS.npz,readonly \
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

Quick smoke/short run example:

```bash
python3 main.py \
  --config ./config/nturgbd-cross-subject/default.yaml \
  --work-dir ../work_dir/ctrgcn_ntu60_xsub_joint_smoke \
  --num-epoch 1 \
  --batch-size 8 \
  --test-batch-size 8 \
  --num-worker 0
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

# 🚀 Build & Run: Baselines

Baseline scripts (MLP sanity checks, tiny ST-GCN, synthetic benchmarks) live in `baselines/`
and run off the Kaggle NPZ or synthetic data.

### Build image

```bash
docker build -t skeleton-lab:baselines -f baselines.docker .
```

### Run container (mount Kaggle NPZ)

```bash
docker run -it --rm --gpus all \
  --shm-size=8g \
  --mount type=bind,source="$HOME/Datasets/NTU60/kaggle_raw/NTU60_CS.npz",target=/workspace/data/NTU60/NTU60_CS.npz,readonly \
  skeleton-lab:baselines
```

Inside the container:

```bash
# GPU sanity
python /workspace/baselines/check_gpu.py

# MLP sanity run on Kaggle NPZ
python /workspace/baselines/train_npz_mlp.py

# Synthetic model timings
python /workspace/baselines/benchmark_models.py --steps 50 --device cuda:0 --amp
```

Notes:
- `train_skeleton_gcn.py` / `train_skeleton_gcn_simple_backbone.py` and the shared dataset
  now accept Kaggle or CTR layouts directly (the loader reshapes Kaggle `(N, T, 150)` to
  `(N, 3, T, 25, 2)` and converts one-hot labels).
- CTR-GCN–specific training loops were moved to `ctrgcn_extras/` and are not part of the
  baselines image.

# 🚀 Build & Run: MS-G3D

MS-G3D now trains directly from the Kaggle NPZ (no MS-G3D `xsub` layout needed).

### Build image

```bash
docker build -t skeleton-lab:msg3d -f msg3d.docker .
```

### Run container (mount Kaggle NPZ)

```bash
docker run -it --rm --gpus all \
  --shm-size=8g \
  --mount type=bind,source="$HOME/Datasets/NTU60/kaggle_raw/NTU60_CS.npz",target=/workspace/MS-G3D/data/ntu/NTU60_CS.npz,readonly \
  skeleton-lab:msg3d
```

### Example training command (cross-subject Kaggle split)

```bash
cd /workspace/MS-G3D
python3 main.py \
  --config config/nturgbd-cross-subject/train_joint_kaggle.yaml \
  --work-dir /workspace/work_dir/ntu60_xsub_msg3d_joint_kaggle \
  --device 0 \
  --num-worker 4
```

> ℹ️ The upstream MS-G3D code uses NVIDIA Apex for mixed precision (`--half` flag), but
> Apex is not bundled in this image. Run without `--half`, or install Apex in the container
> if you need it.

> ℹ️ If you see DataLoader worker bus errors or `/dev/shm` “No space left on device”, run
> the container with `--shm-size=8g` (as shown above) and reduce loader workers via
> `--num-worker 4` (or lower).

> ℹ️ If you hit GPU OOM, use `--memory-friendly` (sets smaller batch/window/workers) or
> manually lower `--batch-size`, `--forward-batch-size`, and set `--train-window-size 64`
> (and `--test-window-size 64`), plus fewer workers.

### Quick smoke test (short run)

For a fast sanity check (1 epoch, small batch/window), use:

```bash
cd /workspace/MS-G3D
python3 main.py \
  --config config/nturgbd-cross-subject/train_joint_kaggle.yaml \
  --work-dir /workspace/work_dir/ntu60_xsub_msg3d_joint_smoke \
  --device 0 \
  --num-epoch 1 \
  --batch-size 8 \
  --forward-batch-size 4 \
  --train-window-size 64 \
  --test-window-size 64 \
  --num-worker 0 \
  --memory-friendly \
  --eval-interval 1 \
  --save-interval 1
```

Feel free to shrink further for quicker runs: e.g., `--batch-size 4 --forward-batch-size 2`
and `--train-window-size 48` (or 32) if VRAM is tight.

---
# 🚀 Build & Run: FreqMixFormer

FreqMixFormer can read either the Kaggle `(N, T, 150)` NTU60 file or a CTR-GCN-path symlink of that file. Point the config to whichever path you mount.

### Build image

```bash
docker build -t skeleton-lab:freqmixformer -f freqmixformer.docker .
```

This image installs the bundled `torchlight` package from `FreqMixFormer/torchlight` so `DictAction` and logging utilities resolve correctly, and includes `h5py` for torchlight’s IO helpers plus `einops`, `torch-dct`, and `scikit-optimize` (used by `ensemble.py`).

### Run container

```bash
docker run -it --rm --gpus all \
  --shm-size=8g \
  --mount type=bind,source="$HOME/Datasets/NTU60",target=/workspace/data/NTU60,readonly \
  skeleton-lab:freqmixformer
```

### Example training command (cross-subject, joint stream)

Inside the container:

```bash
cd /workspace/FreqMixFormer
python3 main.py \
  --config config/nturgbd-cross-subject/joint.yaml \
  --work-dir /workspace/work_dir/freqmixformer_ntu60_xsub_joint \
  --device 0
```

Use `config/nturgbd-cross-subject/motion.yaml` to train the motion stream. To feed the Kaggle one-hot file directly, change both `data_path` entries in the chosen config to `/workspace/data/NTU60/kaggle_raw/NTU60_CS.npz`; the loader auto-detects layout and label format.

**Low-memory option:** If you encounter GPU OOM, a lighter config is provided:

```bash
cd /workspace/FreqMixFormer
python3 main.py \
  --config config/nturgbd-cross-subject/joint_lowmem.yaml \
  --work-dir /workspace/work_dir/freqmixformer_ntu60_xsub_joint_lowmem \
  --device 0
```

This uses `window_size=48`, disables random rotation, sets batch/test batch to 32, and limits data loader workers to 4.

For the motion stream, use the analogous preset:

```bash
python3 main.py \
  --config config/nturgbd-cross-subject/motion_lowmem.yaml \
  --work-dir /workspace/work_dir/freqmixformer_ntu60_xsub_motion_lowmem \
  --device 0
```

---
# 🚀 Build & Run: SkateFormer

SkateFormer consumes the Kaggle-format NTU60 archive (`NTU60_CS.npz` with `x_train/x_test/y_train/y_test`) and aligns with the same PyTorch 2.3 / CUDA 12.1 base as the other stacks.

### Build image

```bash
docker build -t skeleton-lab:skateformer -f skateformer.docker .
```

### Run container

Mount the Kaggle NTU60 file directly into the expected path under `SkateFormer/data/ntu`:

```bash
docker run -it --rm --gpus all \
  --shm-size=8g \
  --mount type=bind,source="$HOME/Datasets/NTU60/kaggle_raw/NTU60_CS.npz",target=/workspace/SkateFormer/data/ntu/NTU60_CS.npz,readonly \
  skeleton-lab:skateformer
```

### Example training command (cross-subject, joint modality)

Inside the container:

```bash
cd /workspace/SkateFormer
python3 main.py \
  --config ./config/train/ntu_cs/SkateFormer_j.yaml \
  --work-dir /workspace/work_dir/skateformer_ntu60_xsub_joint \
  --device 0
```

For the bone modality, swap to `./config/train/ntu_cs/SkateFormer_b.yaml`. Other upstream configs (NTU-Inter, NTU120, NW-UCLA) are available if you mount the corresponding raw datasets in their expected layouts under `/workspace/SkateFormer/data/`.

> ℹ️ SkateFormer expects the Kaggle layout (flattened joints + one-hot labels) by default; no conversion to the CTR-GCN tensor is required for NTU60.

### SkateFormer presets and tips

- Default: batch 128, window 64, heads 32, workers 4 (`SkateFormer_j.yaml`)
- Lowmem: batch 32, window 48, heads 16, workers 0 (`SkateFormer_j_lowmem.yaml`)
- Tiny: batch 8, window 32, heads 8, workers 0, smaller partitions (`SkateFormer_j_tiny.yaml`)

If you see CUDA OOM, shrink the workload in `config/train/ntu_cs/SkateFormer_j.yaml` and the matching test config:

- Lower `batch_size` and `test_batch_size` (e.g., 32 or 16 instead of 128).
- Reduce `window_size` in both `train_feeder_args` and `test_feeder_args` (e.g., 48 or 32).
- Drop DataLoader workers: set `num_worker: 0` to trim host RAM and worker overhead.
- If still tight, trim the model heads: set `num_heads` to 16 or 8 in `model_args` (accuracy may drop).
- Keep `--shm-size=8g` on the container; if shared memory is scarce, lower workers further.

Quick low-memory override example (after editing the YAML as above):

```bash
docker run -it --rm --gpus all --shm-size=8g \
  --mount type=bind,source="$HOME/Datasets/NTU60/kaggle_raw/NTU60_CS.npz",target=/workspace/SkateFormer/data/ntu/NTU60_CS.npz,readonly \
  skeleton-lab:skateformer

# inside
cd /workspace/SkateFormer
python3 main.py \
  --config ./config/train/ntu_cs/SkateFormer_j.yaml \
  --work-dir /workspace/work_dir/skateformer_ntu60_xsub_joint_lowmem \
  --device 0
```

Alternatively, use the prewritten low-memory configs:

```bash
# train (low-memory)
python3 main.py \
  --config ./config/train/ntu_cs/SkateFormer_j_lowmem.yaml \
  --work-dir /workspace/work_dir/skateformer_ntu60_xsub_joint_lowmem \
  --device 0

# test/eval (low-memory)
python3 main.py \
  --config ./config/test/ntu_cs/SkateFormer_j_lowmem.yaml \
  --device 0
```

If VRAM is still tight, use the ultra-small “tiny” configs (batch 8, window 32, heads 8):

```bash
# train (tiny)
python3 main.py \
  --config ./config/train/ntu_cs/SkateFormer_j_tiny.yaml \
  --work-dir /workspace/work_dir/skateformer_ntu60_xsub_joint_tiny \
  --device 0

# test/eval (tiny)
python3 main.py \
  --config ./config/test/ntu_cs/SkateFormer_j_tiny.yaml \
  --device 0
```

The tiny configs also shrink partition sizes (`type_1_size` etc.) to avoid invalid reshapes when the temporal length is very short after cropping/downsampling.

You can also override YAML values directly from the CLI (the parser loads the YAML, then any `--flag` you pass wins). Examples:

```bash
# Use the standard config but override memory-heavy pieces inline
python3 main.py \
  --config ./config/train/ntu_cs/SkateFormer_j.yaml \
  --work-dir /workspace/work_dir/skateformer_custom \
  --batch-size 8 --test-batch-size 8 --num-worker 0 \
  --train-feeder-args "{'window_size':32,'thres':32}" \
  --test-feeder-args "{'window_size':32,'thres':32}" \
  --model-args "{'num_heads':8,'drop_path':0.05,'type_1_size':[4,6],'type_2_size':[4,8],'type_3_size':[4,6],'type_4_size':[4,8]}"

# Fully CLI-driven (no YAML) — verbose but possible
python3 main.py \
  --phase train \
  --work-dir /workspace/work_dir/skateformer_full_cli \
  --feeder feeders.feeder_ntu.Feeder \
  --train-feeder-args "{'data_path':'./data/ntu/NTU60_CS.npz','split':'train','window_size':32,'p_interval':[0.5,0.75],'thres':32,'uniform':True,'partition':True}" \
  --test-feeder-args "{'data_path':'./data/ntu/NTU60_CS.npz','split':'test','window_size':32,'p_interval':[0.95],'thres':32,'uniform':True,'partition':True}" \
  --model model.SkateFormer.SkateFormer_ \
  --model-args "{'num_classes':60,'num_people':2,'num_points':24,'kernel_size':7,'num_heads':8,'drop_path':0.05,'type_1_size':[4,6],'type_2_size':[4,8],'type_3_size':[4,6],'type_4_size':[4,8],'mlp_ratio':4.0,'index_t':True}" \
  --batch-size 8 --test-batch-size 8 --num-worker 0 --num-epoch 500 \
  --optimizer AdamW --base-lr 1e-3 --loss-type LSCE

# Quickly shorten training for smoke tests
python3 main.py \
  --config ./config/train/ntu_cs/SkateFormer_j_tiny.yaml \
  --work-dir /workspace/work_dir/skateformer_ntu60_xsub_joint_tiny_smoke \
  --num-epoch 5 \
  --device 0

# Quick eval (tiny preset; requires a weights path)
python3 main.py \
  --phase test \
  --config ./config/test/ntu_cs/SkateFormer_j_tiny.yaml \
  --weights /workspace/work_dir/skateformer_ntu60_xsub_joint_tiny_smoke/runs-5-XXXX.pt \
  --device 0 \
  --test-batch-size 8
```

Any flag you pass on the CLI overrides the YAML value, so you can mix and match (e.g., `--batch-size 4 --train-feeder-args "{'window_size':24,'thres':24}"`).

> ℹ️ Saved checkpoints are named `runs-{epoch}-{global_step}.pt` (e.g., `runs-1-5011.pt` for a 1-epoch run with 5,011 batches). Adjust the `--weights` path in the test command to the actual filename produced in your `work_dir`.

---

## 🛠 Helper Scripts & Benchmarks

All Python utilities at the repository root are copied into `/workspace/scripts` during the
Docker build for quick access inside containers.

### Dataset tools and sanity checks

| Script | Purpose |
|--------|---------|
| `baselines/check_gpu.py` | Print CUDA availability, device count, device name, and PyTorch/CUDA versions. |
| `dataset_tools/inspect_npz.py` | Dump keys, shapes, and dtypes in an NTU60 `.npz` archive (defaults to `/workspace/data/NTU60/NTU60_CS.npz`). |
| `dataset_tools/convert_kaggle_to_skeleton5d.py` | Convert CTR-layout NPZ to a skeleton5d NPZ for toy GCN scripts. |

### Baselines and experiments

| Script | Purpose |
|--------|---------|
| `baselines/skeleton_dataset_ctrgcn.py` | PyTorch `Dataset` that auto-detects Kaggle or CTR layouts and reshapes accordingly. |
| `baselines/train_npz_mlp.py` | Lightweight MLP baseline on the Kaggle `.npz` file (good VRAM/profiling sanity check). |
| `baselines/train_skeleton_gcn.py` | Toy spatio-temporal baseline built on `skeleton_dataset_ctrgcn.py`. |
| `baselines/train_skeleton_gcn_simple_backbone.py` | Simplified backbone variant for quick iterations. |
| `ctrgcn_extras/train_ctrgcn_npz.py` / `train_ctrgcn_npz_full.py` | CTR-GCN-specific training entry points for converted tensors (outside the baselines image). |

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
