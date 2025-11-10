# train_ctrgcn_npz.py
"""
Train CTR-GCN's Model on the converted NTU60_CS_ctrgcn.npz using our own Dataset
and training loop (with AMP).

We assume:
    - Converted file at: /workspace/CTR-GCN/data/ntu/NTU60_CS_ctrgcn.npz
    - Dataset: NTU60SkeletonDataset (our own)
    - CTR-GCN repo cloned at: /workspace/CTR-GCN

You may need to slightly adjust the import of `Model` depending on the CTR-GCN repo
layout (see comments below).
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from skeleton_dataset_ctrgcn import NTU60SkeletonDataset

# ---------------------------------------------------------------------
# IMPORT CTR-GCN MODEL
# ---------------------------------------------------------------------
# This is a best guess based on common CTR-GCN layouts:
#   - model/ctrgcn.py contains a class `Model`
#   - model/__init__.py may expose it
#
# If this import fails inside the container, run:
#   ls /workspace/CTR-GCN/model
# and adjust accordingly, e.g.:
#   from model.ctrgcn import Model
# or:
#   from CTR-GCN.model.ctrgcn import Model
# depending on PYTHONPATH.
# ---------------------------------------------------------------------
try:
    from model.ctrgcn import Model  # type: ignore
except ImportError as e:
    raise ImportError(
        "Could not import CTR-GCN Model. "
        "Please check /workspace/CTR-GCN/model for the correct module name "
        "and adjust the import in train_ctrgcn_npz.py.\n"
        f"Original error: {e}"
    )


def main():
    npz_path = Path("/workspace/CTR-GCN/data/ntu/NTU60_CS_ctrgcn.npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found at {npz_path}")

    # Dataset & loader
    train_ds = NTU60SkeletonDataset(
        npz_path=npz_path,
        split="train",
        use_both_persons=True,  # or False if you want single-person only
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

    # -----------------------------------------------------------------
    # Instantiate CTR-GCN Model
    # -----------------------------------------------------------------
    # Common CTR-GCN Model signature:
    #   Model(num_class, num_point, num_person, graph, graph_args, in_channels=3, ...)
    #
    # For our data:
    #   num_class  = 60
    #   num_point  = 25
    #   num_person = 2
    #
    # `graph` and `graph_args` will depend on the repo; often they use a
    # graph class like "graph.ntu_rgb_d.Graph". We'll start with a minimal
    # setting and you can adjust to match the original configs.
    # -----------------------------------------------------------------
    num_class = 60
    num_point = 25
    num_person = 2
    in_channels = 3

    # Minimal graph config (you may tweak this to match CTR-GCN examples)
    graph_cfg = {
        "layout": "ntu-rgb+d",
        "strategy": "spatial",
    }

    # Many CTR-GCN repos call:
    #   model = Model(num_class=num_class, num_point=num_point, num_person=num_person,
    #                 graph="graph.ntu_rgb_d.Graph", graph_args=graph_cfg, in_channels=in_channels)
    # Adjust if necessary once you inspect the actual Model.__init__ signature.
    model = Model(
        num_class=num_class,
        num_point=num_point,
        num_person=num_person,
        graph="graph.ntu_rgb_d.Graph",
        graph_args=graph_cfg,
        in_channels=in_channels,
    ).to(device)

    # Optimiser & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)

    model.train()
    max_iters = 100  # short sanity run to see if everything wires up

    for i, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=True)  # (B, C, T, V, M)
        y = y.to(device, non_blocking=True)  # (B,)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(x)  # CTR-GCN expects (B, C, T, V, M)-like input
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

