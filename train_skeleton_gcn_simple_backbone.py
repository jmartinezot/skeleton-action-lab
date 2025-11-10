# train_skeleton_gcn.py
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from skeleton_dataset_ctrgcn import NTU60SkeletonDataset


class SimpleSTBackbone(nn.Module):
    """
    Very simple spatio-temporal baseline:
    - Uses first person only: (B, C, T, V)
    - Temporal 1D convs over T
    - Linear over joints

    This is NOT a real ST-GCN, just a cheap baseline to stress the pipeline.
    """
    def __init__(self, num_classes: int = 60, in_channels: int = 3, T: int = 300, V: int = 25):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(9, 1), padding=(4, 0)),  # (B, 64, T, V)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(9, 1), padding=(4, 0)),          # (B, 128, T, V)
            nn.ReLU(inplace=True),
        )
        self.pool_t = nn.AdaptiveAvgPool2d((1, V))  # pool over time -> (B, 128, 1, V)
        self.fc = nn.Sequential(
            nn.Flatten(),                            # (B, 128 * V)
            nn.Linear(128 * V, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # x: (B, C, T, V) or (B, C, T, V, M)
        if x.ndim == 5:
            x = x[..., 0]  # keep first person -> (B, C, T, V)

        x = self.temporal(x)         # (B, 128, T, V)
        x = self.pool_t(x)           # (B, 128, 1, V)
        logits = self.fc(x)          # (B, num_classes)
        return logits


def main():
    npz_path = Path("/workspace/CTR-GCN/data/ntu/NTU60_CS_ctrgcn.npz")
    ds = NTU60SkeletonDataset(npz_path, split="train", use_both_persons=True)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("Using device:", device)
    print("Mixed precision (AMP) enabled:", use_amp)

    model = SimpleSTBackbone(num_classes=60).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    scaler = GradScaler(enabled=use_amp)

    model.train()
    max_iters = 100  # just a short test run

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optim.zero_grad()

        # Forward pass with autocast if AMP is enabled
        with autocast(enabled=use_amp):
            logits = model(x)
            loss = crit(logits, y)

        # Backward + step with GradScaler if AMP is enabled
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

    print("Finished SimpleSTBackbone AMP sanity training.")


if __name__ == "__main__":
    main()

