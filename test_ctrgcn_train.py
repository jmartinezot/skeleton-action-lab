# test_ctrgcn_train.py
import subprocess
from pathlib import Path

# path to your CTR-GCN repo
repo = Path("/workspace/CTR-GCN")
config = repo / "config/nturgbd-cross-subject/train_joint.yaml"
workdir = Path("/workspace/work_dir/ctrgcn_test")

if not config.exists():
    raise FileNotFoundError(f"Config not found: {config}")

print("Running CTR-GCN with config:", config)
print("Workdir:", workdir)

cmd = [
    "python", "main.py",
    "--config", str(config),
    "--work-dir", str(workdir)
]

subprocess.run(cmd, cwd=repo)



