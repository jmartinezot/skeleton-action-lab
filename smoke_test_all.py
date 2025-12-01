"""
Run all model smoke tests inside their Docker images and report pass/fail.

Example (CPU-only):
    python3 smoke_test_all.py --device cpu

Example (GPU, mount Kaggle NPZ for Hyper-GCN):
    python3 smoke_test_all.py --device cuda:0 --use-gpus \\
        --hyper-data /home/datasets/NTU60/kaggle_raw/NTU60_CS.npz
"""

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class SmokeTest:
    name: str
    image: str
    workdir: str
    cmd: List[str]
    mounts: List[Tuple[Path, str, bool]] = field(default_factory=list)  # (host, container, readonly)


def docker_available() -> bool:
    return shutil.which("docker") is not None


def build_docker_cmd(test: SmokeTest, use_gpus: bool) -> List[str]:
    cmd = ["docker", "run", "--rm", "--shm-size=8g"]
    if use_gpus:
        cmd.extend(["--gpus", "all"])
    for host, container, readonly in test.mounts:
        opts = f"type=bind,source={host},target={container}"
        if readonly:
            opts += ",readonly"
        cmd.extend(["--mount", opts])
    cmd.append(test.image)
    # Ensure we run inside the right working directory
    cmd.extend(["bash", "-lc", f"cd {test.workdir} && " + " ".join(test.cmd)])
    return cmd


def run_test(test: SmokeTest, use_gpus: bool) -> Tuple[bool, str, str, float]:
    docker_cmd = build_docker_cmd(test, use_gpus)
    start = time.time()
    proc = subprocess.run(docker_cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    return proc.returncode == 0, proc.stdout, proc.stderr, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run all Skeleton Action Lab smoke tests (Docker-based)")
    parser.add_argument("--device", default="cpu", help="torch device to pass into each smoke test, e.g., cpu or cuda:0")
    parser.add_argument("--use-gpus", action="store_true", help="add --gpus all to docker run")
    parser.add_argument(
        "--hyper-data",
        type=Path,
        default=None,
        help="Host path to NTU60 Kaggle NPZ for Hyper-GCN smoke test (otherwise runs synthetic)",
    )
    parser.add_argument("--hyper-window", type=int, default=32, help="Temporal length for Hyper-GCN smoke test")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure")
    args = parser.parse_args()

    if not docker_available():
        print("docker not found on PATH; please install Docker and retry.", file=sys.stderr)
        sys.exit(1)

    # Base tests (all use synthetic data unless you mount the Kaggle NPZ for Hyper-GCN)
    tests: List[SmokeTest] = [
        SmokeTest("Baselines", "skeleton-lab:baselines", "/workspace/baselines", ["python3", "smoke_test.py", "--device", args.device]),
        SmokeTest("CTR-GCN", "skeleton-lab:ctrgcn", "/workspace/CTR-GCN", ["python3", "smoke_test.py", "--device", args.device]),
        SmokeTest("MS-G3D", "skeleton-lab:msg3d", "/workspace/MS-G3D", ["python3", "smoke_test.py", "--device", args.device]),
        SmokeTest("FreqMixFormer", "skeleton-lab:freqmixformer", "/workspace/FreqMixFormer", ["python3", "smoke_test.py", "--device", args.device]),
        SmokeTest("SkateFormer", "skeleton-lab:skateformer", "/workspace/SkateFormer", ["python3", "smoke_test.py", "--device", args.device]),
        SmokeTest(
            "Hyper-GCN",
            "skeleton-lab:hypergcn",
            "/workspace/Hyper-GCN",
            ["python3", "smoke_test_kaggle.py", "--device", args.device, "--window-size", str(args.hyper_window)],
        ),
        SmokeTest("FS-VAE", "skeleton-lab:fsvae", "/workspace/FS-VAE", ["python3", "smoke_test.py", "--device", args.device]),
        SmokeTest("MSF-GZSSAR", "skeleton-lab:msf-gzssar", "/workspace/MSF-GZSSAR", ["python3", "smoke_test.py", "--device", args.device]),
    ]

    # Optional Hyper-GCN data mount
    if args.hyper_data:
        hyper_host = args.hyper_data.expanduser().resolve()
        tests[5].mounts.append((hyper_host, "/workspace/Hyper-GCN/data/NTU60_CS.npz", True))
        tests[5].cmd.extend(["--data-path", "/workspace/Hyper-GCN/data/NTU60_CS.npz"])
    else:
        tests[5].cmd.append("--use-synthetic")

    results = []
    print(f"Running {len(tests)} smoke tests in Docker...\n")
    for idx, test in enumerate(tests, 1):
        print(f"[{idx}/{len(tests)}] {test.name} ", end="", flush=True)
        ok, stdout, stderr, elapsed = run_test(test, args.use_gpus)
        status = "OK" if ok else "FAIL"
        print(f"{status} ({elapsed:.1f}s)")
        if stdout.strip():
            print(stdout.strip())
        if not ok and stderr.strip():
            print(stderr.strip(), file=sys.stderr)
        results.append((test.name, ok))
        if not ok and args.fail_fast:
            break

    failed = [name for name, ok in results if not ok]
    if failed:
        print("\nSmoke suite completed with failures: " + ", ".join(failed))
        print("If a container reports missing smoke_test.py, rebuild that image so the new script is included.")
        sys.exit(1)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
