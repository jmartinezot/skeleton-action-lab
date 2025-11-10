# train_ctrgcn_npz_full.py
"""
Full CTR-GCN training loop on NTU60_CS_ctrgcn.npz:
- train on x_train / y_train
- evaluate on x_test / y_test each epoch
- uses AMP when CUDA is available
"""

from pathlib import Path
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from skeleton_dataset_ctrgcn import NTU60SkeletonDataset

# ---------------------------------------------------------------------
# Ensure CTR-GCN root is FIRST on sys.path so its `model` and `graph`
# packages shadow MS-G3D's.
# ---------------------------------------------------------------------
CTR_GCN_ROOT = "/workspace/CTR-GCN"
MS_G3D_ROOT = "/workspace/MS-G3D"

if CTR_GCN_ROOT in sys.path:
    sys.path.remove(CTR_GCN_ROOT)
if MS_G3D_ROOT in sys.path:
    sys.path.remove(MS_G3D_ROOT)

sys.path.insert(0, CTR_GCN_ROOT)
sys.path.append(MS_G3D_ROOT)

try:
    from model import ctrgcn  # /workspace/CTR-GCN/model/ctrgcn.py
    Model = ctrgcn.Model
except Exception as e:
    raise ImportError(
        "Could not import CTR-GCN Model from /workspace/CTR-GCN/model/ctrgcn.py.\n"
        f"Original error: {e}"
    )


@dataclass
class Config:
    npz_path: Path = Path("/workspace/CTR-GCN/data/ntu/NTU60_CS_ctrgcn.npz")
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 10
    lr: float = 1e-3
    num_class: int = 60
    num_point: int = 25
    num_person: int = 2
    in_channels: int = 3


def make_loaders(cfg: Config):
    train_ds = NTU60SkeletonDataset(
        npz_path=cfg.npz_path,
        split="train",
        use_both_persons=True,
    )
    test_ds = NTU60SkeletonDataset(
        npz_path=cfg.npz_path,
        split="test",
        use_both_persons=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(total, 1)


def main():
    cfg = Config()

    if not cfg.npz_path.exists():
        raise FileNotFoundError(f"NPZ not found at {cfg.npz_path}")

    train_loader, test_loader = make_loaders(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("Using device:", device)
    print("Mixed precision (AMP) enabled:", use_amp)

    graph_cfg = {"labeling_mode": "spatial"}

    model = Model(
        num_class=cfg.num_class,
        num_point=cfg.num_point,
        num_person=cfg.num_person,
        graph="graph.ntu_rgb_d.Graph",
        graph_args=graph_cfg,
        in_channels=cfg.in_channels,
        drop_out=0.0,
        adaptive=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n=== Epoch {epoch}/{cfg.epochs} ===")
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            running_correct += (pred == y).sum().item()
            running_total += y.numel()

            if i % 50 == 0:
                batch_acc = (pred == y).float().mean().item()
                print(
                    f"[epoch {epoch} iter {i}] "
                    f"loss={loss.item():.4f}, "
                    f"batch_acc={batch_acc:.3f}"
                )

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch} summary: "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.3f}, "
            f"test_acc={test_acc:.3f}"
        )

    print("\nFinished full CTR-GCN training on NPZ.")


if __name__ == "__main__":
    main()

