#!/usr/bin/env python3
import argparse
import numpy as np
import pickle
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert CTR-GCN NPZ layout into MS-G3D xsub splits."
    )
    parser.add_argument(
        "--input",
        default="/workspace/data/NTU60/ctrgcn/NTU60_CS.npz",
        help="Path to CTR-GCN-style NTU60_CS.npz (defaults to container mount)",
    )
    parser.add_argument(
        "--output-root",
        default="/workspace/MS-G3D/data/ntu",
        help="Output root where xsub/ will be created",
    )
    return parser.parse_args()


args = parse_args()
src = Path(args.input).expanduser()
out = Path(args.output_root).expanduser() / "xsub"
out.mkdir(parents=True, exist_ok=True)

if not src.is_file():
    raise FileNotFoundError(f"Input NPZ not found: {src}")

npz = np.load(src, allow_pickle=True)
print("Loaded:", src)
print("Keys:", npz.files)

x_tr = npz["x_train"]  # (N, 300, 150)
y_tr = npz["y_train"]  # (N, 60) one-hot
x_te = npz["x_test"]
y_te = npz["y_test"]

def convert_x(x):
    N, T, F = x.shape
    assert F == 150, f"Expected last dim 150, got {F}"
    # reshape (N,T,150)->(N,T,3,25,2) then permute to (N,3,T,25,2)
    x = x.reshape(N, T, 3, 25, 2).transpose(0, 2, 1, 3, 4).astype(np.float32)
    return x

def onehot_to_index(y):
    # y is (N, 60) float; get argmax along classes
    idx = np.argmax(y, axis=1).astype(np.int64)
    return idx

x_tr_out = convert_x(x_tr)
x_te_out = convert_x(x_te)
y_tr_idx = onehot_to_index(y_tr)
y_te_idx = onehot_to_index(y_te)

# Validate label range and auto-fix 1-based labels
num_classes = y_tr.shape[1]
min_label = int(min(y_tr_idx.min(), y_te_idx.min()))
max_label = int(max(y_tr_idx.max(), y_te_idx.max()))

if min_label >= 1 and max_label == num_classes:
    print("Detected 1-based labels; shifting to 0-based indexing.")
    y_tr_idx = y_tr_idx - 1
    y_te_idx = y_te_idx - 1
    min_label -= 1
    max_label -= 1
elif max_label >= num_classes or min_label < 0:
    raise ValueError(
        f"Label indices out of range after conversion. "
        f"Min: {min_label}, Max: {max_label}, Num classes: {num_classes}"
    )

print(f"Label range: min={min_label}, max={max_label}, classes={num_classes}")

print("Converted shapes:")
print("train_data:", x_tr_out.shape, x_tr_out.dtype)  # (N,3,300,25,2)
print("val_data:  ", x_te_out.shape, x_te_out.dtype)
print("train_labels:", y_tr_idx.shape, y_tr_idx.dtype)
print("val_labels:  ", y_te_idx.shape, y_te_idx.dtype)

# Save npy
np.save(out / "train_data.npy", x_tr_out)
np.save(out / "val_data.npy",   x_te_out)
# Also save with the joint naming MS-G3D configs expect
np.save(out / "train_data_joint.npy", x_tr_out)
np.save(out / "val_data_joint.npy",   x_te_out)

# Save labels as MS-G3D expects: (sample_name_list, labels_list)
train_sample_names = [str(i) for i in range(len(y_tr_idx))]
val_sample_names = [str(i) for i in range(len(y_te_idx))]

with open(out / "train_label.pkl", "wb") as f:
    pickle.dump((train_sample_names, list(map(int, y_tr_idx))), f)
with open(out / "val_label.pkl", "wb") as f:
    pickle.dump((val_sample_names, list(map(int, y_te_idx))), f)

print(f"✅ Wrote: {out}")
