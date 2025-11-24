"""
Convert Kaggle NTU60_CS.npz format (N, T, D=150) with one-hot labels
into CTR-GCN-style skeleton tensors:

    x_*: (N, C=3, T, V=25, M=2)
    y_*: (N,) integer labels in [0, 59)

Usage:
    python3 convert_ntu60_kaggle_to_ctrgcn.py \
        --input PATH_TO_INPUT.npz \
        --output PATH_TO_OUTPUT.npz
"""

import argparse
from pathlib import Path
import numpy as np


def reshape_x(x):
    """
    x: (N, T, D) with D = 150 = 25*3*2
    return: (N, C=3, T, V=25, M=2)
    """
    n, t, d = x.shape
    if d != 150:
        raise ValueError(f"Expected D=150 (25*3*2), got D={d}")

    v = 25  # joints
    c = 3   # coords
    m = 2   # persons

    # (N, T, 150) -> (N, T, V, C, M)
    x = x.reshape(n, t, v, c, m)
    # -> (N, C, T, V, M)
    return np.transpose(x, (0, 3, 1, 2, 4))


def convert_labels(y):
    """
    y: (N, num_classes) one-hot
    return: (N,) integer labels
    """
    if y.ndim != 2:
        raise ValueError(f"Expected y to be 2D (one-hot), got {y.shape}")
    return y.argmax(axis=1).astype(np.int64)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Kaggle NTU60 npz to CTR-GCN 5D layout")
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to input Kaggle-style NTU60_CS.npz",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to save converted CTR-GCN-style npz file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input NPZ not found: {in_path}")

    print(f"Loading {in_path} ...")
    data = np.load(in_path, allow_pickle=True)

    required = ["x_train", "y_train", "x_test", "y_test"]
    for k in required:
        if k not in data.files:
            raise KeyError(f"Missing key '{k}' in input file. Found keys: {data.files}")

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    print("Original shapes:")
    print("  x_train:", x_train.shape, x_train.dtype)
    print("  y_train:", y_train.shape, y_train.dtype)
    print("  x_test :", x_test.shape, x_test.dtype)
    print("  y_test :", y_test.shape, y_test.dtype)

    print("\nReshaping x to (N, C=3, T, V=25, M=2) ...")
    x_train_conv = reshape_x(x_train)
    x_test_conv = reshape_x(x_test)

    print("New x_train:", x_train_conv.shape)
    print("New x_test :", x_test_conv.shape)

    print("\nConverting y from one-hot to class indices ...")
    y_train_idx = convert_l_


