# inspect_npz.py
import sys
import numpy as np
from pathlib import Path

# Allow path override via command-line argument
path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/ntu/NTU60_CS.npz")

if not path.exists():
    raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

data = np.load(path, allow_pickle=True)
print(f"Loaded {path.name}")
print("Keys:", data.files)
print()

for key in data.files:
    arr = data[key]
    print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")



