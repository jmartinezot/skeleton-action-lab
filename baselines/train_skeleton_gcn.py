# train_skeleton_gcn.py
from pathlib import Path
import argparse
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from skeleton_dataset_ctrgcn import NTU60SkeletonDataset
from stgcn_backbone import STGCNBackbone


def resolve_device(device_arg: str | None) -> torch.device:
    """
    Accepts: cpu, cuda, cuda:0, cuda:1, ...
    Falls back automatically with warnings.
    """
    def auto_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg in (None, "", "auto"):
        return auto_device()

    try:
        dev = torch.device(device_arg)
    except Exception as exc:
        print(f"[train_skeleton_gcn] Invalid device '{device_arg}' ({exc}); using auto.", file=sys.stderr)
        return auto_device()

    if dev.type == "cuda":
        if not torch.cuda.is_available():
            print(f"[train_skeleton_gcn] CUDA requested but not available; using CPU.", file=sys.stderr)
            return torch.device("cpu")
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            print(
                f"[train_skeleton_gcn] GPU index {dev.index} out of range "
                f"(found {torch.cuda.device_count()} GPUs); using cuda:0.",
                file=sys.stderr,
            )
            return torch.device("cuda:0")

    return dev


def parse_args():
    parser = argparse.ArgumentParser(description="Train ST-GCN sanity check on NTU60 NPZ (Kaggle or CTR layout)")
    parser.add_argument(
        "--npz-path",
        type=str,
        default="/workspace/data/NTU60/NTU60_CS.npz",
        help="Path to NTU60_CS.npz (Kaggle one-hot) or CTR-style file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs (dataset passes)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cpu, cuda, cuda:0, cuda:1, ... (default: auto)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Input npz file not found: {npz_path}")

    ds = NTU60SkeletonDataset(npz_path, split="train", use_both_persons=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    device = resolve_device(args.device)
    use_amp = device.type == "cuda"

    print("Dataset path:", npz_path)
    print("Device:", device)
    print("Mixed precision AMP:", use_amp)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)

    model = STGCNBackbone(num_classes=60, in_channels=3, num_joints=25).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)

    model.train()

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")

        for i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optim.zero_grad()

            with autocast(enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if i % 10 == 0:
                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    acc = (pred == y).float().mean().item()
                print(f"[iter {i}] loss={loss.item():.4f}, acc={acc:.3f}")

    print("\nFinished STGCNBackbone AMP training.")


if __name__ == "__main__":
    main()


