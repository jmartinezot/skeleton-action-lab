# train_npz_mlp.py
"""
Minimal sanity check training on NTU60_CS.npz

- Loads x_train, y_train from the Kaggle NPZ (T=300, D=150, one-hot labels)
- Flattens (T, D) -> (T*D) and runs a small MLP
- Trains for a few batches just to stress GPU & VRAM
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import argparse


class NTU60NPZDataset(Dataset):
    def __init__(self, npz_path: Path, split: str = "train"):
        data = np.load(npz_path, allow_pickle=True)
        if split == "train":
            self.x = data["x_train"]  # (N, T, D)
            self.y = data["y_train"]  # (N, 60) one-hot
        else:
            self.x = data["x_test"]
            self.y = data["y_test"]

        # convert one-hot to class indices
        self.y = self.y.argmax(axis=1).astype(np.int64)

        assert self.x.ndim == 3, f"Expected x to be (N, T, D), got {self.x.shape}"
        assert self.y.ndim == 1, f"Expected y to be (N,), got {self.y.shape}"

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]   # (T, D)
        y = self.y[idx]   # scalar class index
        return x.astype("float32"), int(y)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 60):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # x: (B, T, D)
        b, t, d = x.shape
        x = x.view(b, t * d)  # flatten
        return self.net(x)


def run(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found at {npz_path}")

    print(f"Loading dataset from {npz_path} ...")
    train_ds = NTU60NPZDataset(npz_path, split="train")
    print(f"Train size: {len(train_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    sample_x, _ = train_ds[0]
    T, D = sample_x.shape
    model = SimpleMLP(input_dim=T * D, num_classes=60).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    max_batches = 50
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            _, pred = logits.max(1)
            acc = (pred == y).float().mean().item()
            print(f"[{batch_idx}/{max_batches}] loss={loss.item():.4f}, acc={acc:.3f}")

        if batch_idx >= max_batches:
            break

    print("Finished sanity-check training loop.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny MLP sanity check on Kaggle NTU60 NPZ")
    parser.add_argument(
        "--npz-path",
        type=str,
        default="/workspace/data/NTU60/NTU60_CS.npz",
        help="Path to Kaggle NTU60_CS.npz",
    )
    args = parser.parse_args()
    run(Path(args.npz_path))
