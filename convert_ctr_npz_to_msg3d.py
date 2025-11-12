#!/usr/bin/env python3
import numpy as np
import pickle
from pathlib import Path

src = Path("/workspace/MS-G3D/data/ntu/NTU60_CS.npz")
out = Path("/workspace/MS-G3D/data/ntu/xsub")
out.mkdir(parents=True, exist_ok=True)

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

print("Converted shapes:")
print("train_data:", x_tr_out.shape, x_tr_out.dtype)  # (N,3,300,25,2)
print("val_data:  ", x_te_out.shape, x_te_out.dtype)
print("train_labels:", y_tr_idx.shape, y_tr_idx.dtype)
print("val_labels:  ", y_te_idx.shape, y_te_idx.dtype)

# Save npy
np.save(out / "train_data.npy", x_tr_out)
np.save(out / "val_data.npy",   x_te_out)

# Save labels as MS-G3D expects: (labels_list, sample_name_list)
with open(out / "train_label.pkl", "wb") as f:
    pickle.dump((list(map(int, y_tr_idx)), list(range(len(y_tr_idx)))), f)
with open(out / "val_label.pkl", "wb") as f:
    pickle.dump((list(map(int, y_te_idx)), list(range(len(y_te_idx)))), f)

print(f"✅ Wrote: {out}")



