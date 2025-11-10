# train_ctrgcn_npz.py
"""
Train CTR-GCN's Model on the converted NTU60_CS_ctrgcn.npz using our own Dataset
and training loop (with AMP).

We assume:
    - Converted file at: /workspace/CTR-GCN/data/ntu/NTU60_CS_ctrgcn.npz
    - Dataset: NTU60SkeletonDataset (our own)
    - CTR-GCN repo cloned at: /workspace/CTR-GCN
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from skeleton_dataset_ctrgcn import NTU60SkeletonDataset

# ---------------------------------------------------------------------
# Make sure CTR-GCN's *model directory* is on sys.path
# so we can import ctrgcn.py as a top-level module (no name clash with MS-G3D)
# ---------------------------------------------------------------------
CTR_GCN_ROOT_DIR = Path("/workspace/CTR-GCN")
CTR_GCN_MODEL_DIR = CTR_GCN_ROOT_DIR / "model"

# Add both the repository root (for modules like ``graph.*``) and the ``model``
# subdirectory (for ``ctrgcn.py`` itself) so that CTR-GCN's dynamic imports work
# exactly as they do in the original project.
for path in (CTR_GCN_MODEL_DIR, CTR_GCN_ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    import ctrgcn  # this is /workspace/CTR-GCN/model/ctrgcn.py
    Model = ctrgcn.Model  # type: ignore[attr-defined]
except Exception as e:
    raise ImportError(
        "Could not import CTR-GCN Model from /workspace/CTR-GCN/model/ctrgcn.py.\n"
        "Check inside the container with:\n"
        "    ls /workspace/CTR-GCN/model\n"
        "and confirm that ctrgcn.py defines a class named Model.\n"
        f"Original error: {e}"
    )


def main():
    npz_path = Path("/workspace/CTR-GCN/data/ntu/NTU60_CS_ctrgcn.npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found at {npz_path}")

    # --------------------------- Dataset & loader ---------------------------
    train_ds = NTU60SkeletonDataset(
        npz_path=npz_path,
        split="train",
        use_both_persons=True,  # keep both persons, shape: (B, 3, T, 25, 2)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("Using device:", device)
    print("Mixed precision (AMP) enabled:", use_amp)

    # --------------------------- CTR-GCN Model ---------------------------
    # Common CTR-GCN Model signature:
    #   Model(num_class, num_point, num_person,
    #         graph, graph_args, in_channels=3, drop_out=0, adaptive=True, ...)
    #
    num_class = 60
    num_point = 25
    num_person = 2
    in_channels = 3

    graph_cfg = {
        "layout": "ntu-rgb+d",
        "strategy": "spatial",
    }

    model = Model(
        num_class=num_class,
        num_point=num_point,
        num_person=num_person,
        graph="graph.ntu_rgb_d.Graph",
        graph_args=graph_cfg,
        in_channels=in_channels,
        drop_out=0.0,
        adaptive=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)

    model.train()
    max_iters = 100  # short sanity run

    for i, (x, y) in enumerate(train_loader):
        # x: (B, 3, T, 25, 2)
        # y: (B,)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i % 10 == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().mean().item()
            print(f"[{i}/{max_iters}] loss={loss.item():.4f}, acc={acc:.3f}")

        if i >= max_iters:
            break

    print("Finished CTR-GCN Model AMP sanity training.")


if __name__ == "__main__":
    main()



