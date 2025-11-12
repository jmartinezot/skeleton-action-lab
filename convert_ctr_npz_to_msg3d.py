#!/usr/bin/env python3
import numpy as np
import pickle
from pathlib import Path

# ------------------------------------------------------------
# 1) Paths inside container
# ------------------------------------------------------------
src = Path("/workspace/MS-G3D/data/ntu/NTU60_CS.npz")
out = Path("/workspace/MS-G3D/data/ntu/xsub")
out.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2) Load .npz
# ------------------------------------------------------------
npz = np.load(src, allow_pickle=True)
print("Loaded:", src)
print("Keys:", npz.files)

# ------------------------------------------------------------
# 3) Identify arrays (adjust keys if necessary)
# ------------------------------------------------------------
x_train = npz.get("x_train") or npz.get("train_data") or npz.get("x")
y_train = npz.get("y_train") or npz.get("train_label") or npz.get("y")
x_test  = npz.get("x_test")  or npz.get("val_data")   or npz.get("x_val")
y_test  = npz.get("y_test")  or npz.get("val_label")  or npz.get("y_val")

if None in (x_train, y_train, x_test, y_test):
    raise KeyError(f"Couldn't find all arrays in {src}. Keys: {npz.files}")

print(f"Train data shape: {x_train.shape}, labels: {len(y_train)}")
print(f"Test  data shape: {x_test.shape},  labels: {len(y_test)}")

# ------------------------------------------------------------
# 4) Save in MS-G3D expected structure
# ------------------------------------------------------------
np.save(out / "train_data.npy", x_train)
with open(out / "train_label.pkl", "wb") as f:
    pickle.dump((list(y_train), list(range(len(y_train)))), f)

np.save(out / "val_data.npy", x_test)
with open(out / "val_label.pkl", "wb") as f:
    pickle.dump((list(y_test), list(range(len(y_test)))), f)

print(f"✅ Wrote converted dataset to {out}")
print("Files:", [p.name for p in out.glob('*')])


