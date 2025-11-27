# skeleton_dataset_ctrgcn.py
"""
PyTorch Dataset for NTU60 skeleton data stored in either the Kaggle NPZ or
the CTR-GCN-style NPZ.

Accepted layouts:
- Kaggle: x_* shape (N, T, 150) with one-hot y_* (N, 60)
- CTR-GCN: x_* shape (N, 3, T, 25, 2) with integer y_* (N,)

Outputs:
- x: torch.FloatTensor, shape (C=3, T, V=25, M=2)
- y: torch.LongTensor, scalar class index
"""

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


Split = Literal["train", "test"]


class NTU60SkeletonDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        split: Split = "train",
        use_both_persons: bool = True,
        transform: Optional[callable] = None,
    ):
        """
        Args:
            npz_path: Path to NTU60_CS.npz (Kaggle layout) or CTR layout
            split: "train" or "test"
            use_both_persons:
                - If True: keep M=2 persons as-is (shape: C, T, V, M).
                - If False: only keep the first person (M=1, last dim squeezed).
            transform: Optional callable applied to x (after numpy->torch).
        """
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ not found: {self.npz_path}")

        data = np.load(self.npz_path, allow_pickle=True)

        if split == "train":
            x = data["x_train"]
            y = data["y_train"]
        elif split == "test":
            x = data["x_test"]
            y = data["y_test"]
        else:
            raise ValueError(f"Unknown split: {split}, expected 'train' or 'test'")

        # Accept Kaggle (N, T, 150) or CTR layout (N, 3, T, 25, 2)
        if x.ndim == 3 and x.shape[-1] == 150:
            x = x.reshape(x.shape[0], x.shape[1], 25, 3, 2)            # (N, T, V, C, M)
            x = np.transpose(x, (0, 3, 1, 2, 4))                       # (N, C, T, V, M)
            y = y.argmax(axis=1)
        elif x.ndim == 5:
            pass
        else:
            raise ValueError(f"Unsupported x shape: {x.shape} (expected (N, T, 150) or (N, 3, T, 25, 2))")

        if y.ndim == 2:
            y = y.argmax(axis=1)
        if y.ndim != 1:
            raise ValueError(f"Expected y to be 1D after processing, got {y.shape}")

        self.use_both_persons = use_both_persons
        self.transform = transform
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

        print(
            f"[NTU60SkeletonDataset] Loaded {split} split from {self.npz_path}:\n"
            f"  x: {self.x.shape}, dtype={self.x.dtype}\n"
            f"  y: {self.y.shape}, dtype={self.y.dtype}\n"
            f"  use_both_persons={self.use_both_persons}"
        )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        x = self.x[idx]  # (C, T, V, M)
        y = self.y[idx]  # scalar int

        if not self.use_both_persons:
            # take only first person, shape: (C, T, V)
            x = x[..., 0]  # drop M dimension

        x = torch.from_numpy(x).float()
        y = torch.tensor(int(y), dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)

        return x, y


def _sanity_check():
    """
    Small self-test you can run directly:
        python skeleton_dataset_ctrgcn.py
    """
    npz_path = Path("/workspace/data/NTU60/NTU60_CS.npz")
    ds = NTU60SkeletonDataset(npz_path, split="train", use_both_persons=True)

    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    x, y = next(iter(loader))
    print("\nBatch shapes:")
    print("  x:", x.shape)  # (B, C, T, V, M) or (B, C, T, V) if use_both_persons=False
    print("  y:", y.shape, y.dtype)


if __name__ == "__main__":
    _sanity_check()
