# train_skeleton_gcn.py
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from skeleton_dataset_ctrgcn import NTU60SkeletonDataset
from stgcn_backbone import STGCNBackbone


def main():
    npz_path = Path("/workspace/CTR-GCN/data/ntu/NTU60_CS_ctrgcn.npz")
    ds = NTU60SkeletonDataset(npz_path, split="train", use_both_persons=True)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("Using device:", device)
    print("Mixed precision (AMP) enabled:", use_amp)

    model = STGCNBackbone(num_classes=60, in_channels=3, num_joints=25).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    scaler = GradScaler(enabled=use_amp)

    model.train()
    max_iters = 100  # short sanity run

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
            print(f"[{i}/{max_iters}] loss={loss.item():.4f}, acc={acc:.3f}")

        if i >= max_iters:
            break

    print("Finished STGCNBackbone AMP sanity training.")


if __name__ == "__main__":
    main()

