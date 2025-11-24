# convert_ntu60_kaggle_to_ctrgcn.py
"""
Convert Kaggle NTU60_CS.npz format (N, T, D=150) with one-hot labels
into a more standard skeleton format:

    x_*: (N, C=3, T, V=25, M=2)
    y_*: (N,) integer labels in [0, 59]

Assumptions:
- T = 300, D = 150 = 25 joints * 3 coords * 2 persons
- x is stored as (N, T, D)
- y is stored as (N, num_classes) one-hot

Output:
- NTU60_CS_skel_5d.npz in the same directory as the input.
"""

import numpy as np
from pathlib import Path


def reshape_x(x):
    """
    x: (N, T, D) with D = 150 = 25 * 3 * 2
    return: (N, C=3, T, V=25, M=2)
    """
    n, t, d = x.shape
    if d != 150:
        raise ValueError(f"Expected D=150 (25*3*2), got D={d}")

    v = 25  # joints
    c = 3   # x,y,z
    m = 2   # persons

    # (N, T, 150) -> (N, T, V, C, M)
    x = x.reshape(n, t, v, c, m)
    # -> (N, C, T, V, M)
    x = np.transpose(x, (0, 3, 1, 2, 4))
    return x


def convert_labels(y):
    """
    y: (N, num_classes) one-hot
    return: (N,) integer class indices
    """
    if y.ndim != 2:
        raise ValueError(f"Expected y to be 2D one-hot (N, num_classes), got {y.shape}")
    return y.argmax(axis=1).astype(np.int64)


def main():
    in_path = Path("/workspace/CTR-GCN/data/ntu/NTU60_CS.npz")
    if not in_path.exists():
        raise FileNotFoundError(f"Input NPZ not found at {in_path}")

    print(f"Loading {in_path} ...")
    data = np.load(in_path, allow_pickle=True)

    required_keys = ["x_train", "y_train", "x_test", "y_test"]
    for k in required_keys:
        if k not in data.files:
            raise KeyError(f"Key '{k}' not found in {in_path}. Found keys: {data.files}")

    x_train = data["x_train"]  # (N, T, D)
    y_train = data["y_train"]  # (N, 60) one-hot
    x_test = data["x_test"]
    y_test = data["y_test"]

    print("Original shapes:")
    print("  x_train:", x_train.shape, x_train.dtype)
    print("  y_train:", y_train.shape, y_train.dtype)
    print("  x_test: ", x_test.shape, x_test.dtype)
    print("  y_test: ", y_test.shape, y_test.dtype)

    print("\nReshaping x_* to (N, C=3, T, V=25, M=2) ...")
    x_train_conv = reshape_x(x_train)
    x_test_conv = reshape_x(x_test)

    print("New x_train:", x_train_conv.shape)
    print("New x_test: ", x_test_conv.shape)

    print("\nConverting y_* from one-hot to integer labels ...")
    y_train_idx = convert_labels(y_train)
    y_test_idx = convert_labels(y_test)

    print("New y_train:", y_train_idx.shape, y_train_idx.dtype)
    print("New y_test: ", y_test_idx.shape, y_test_idx.dtype)

    out_path = in_path.with_name("NTU60_CS_skel_5d.npz")
    print(f"\nSaving converted data to {out_path} ...")
    np.savez_compressed(
        out_path,
        x_train=x_train_conv,
        y_train=y_train_idx,
        x_test=x_test_conv,
        y_test=y_test_idx,
    )

    print("Done.")


if __name__ == "__main__":
    main()

