"""Benchmark and smoke-test the available skeleton models.

Run quick synthetic-training loops for one or more models and report
wall-clock timings. The goal is to answer: "Can this environment run the
models, and how long might it take to scale up to bigger runs?" The
benchmarks intentionally use synthetic data so they can execute without the
NTU60 dataset mounted.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


@dataclasses.dataclass
class BenchmarkResult:
    name: str
    success: bool
    seconds: float
    steps: int
    batch_size: int
    seq_len: int
    device: str
    amp: bool
    message: str = ""

    def as_row(self) -> str:
        status = "✅" if self.success else "❌"
        return (
            f"{status} {self.name}: {self.steps} steps, "
            f"batch={self.batch_size}, seq_len={self.seq_len}, device={self.device}, "
            f"amp={self.amp} -> {self.seconds:.2f}s {self.message}".strip()
        )


class BenchmarkTask:
    def __init__(self, name: str, description: str, runner: Callable[[argparse.Namespace], BenchmarkResult]):
        self.name = name
        self.description = description
        self.runner = runner

    def __call__(self, args: argparse.Namespace) -> BenchmarkResult:
        return self.runner(args)


# ---------------------------------------------------------------------------
# Simple ST-GCN-style backbone synthetic training loop
# ---------------------------------------------------------------------------

def run_simple_stgcn(args: argparse.Namespace) -> BenchmarkResult:
    from train_skeleton_gcn_simple_backbone import SimpleSTBackbone

    start = time.perf_counter()
    device = resolve_device(getattr(args, "device", None))
    use_amp = args.amp and device.type == "cuda"

    model = SimpleSTBackbone(num_classes=60, T=args.seq_len).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)

    iterations = args.steps
    batch_size = args.batch_size
    seq_len = args.seq_len

    try:
        for _ in range(iterations):
            # (B, C=3, T, V=25, M=2) -> model keeps first person internally
            x = torch.randn(batch_size, 3, seq_len, 25, 2, device=device)
            y = torch.randint(0, 60, (batch_size,), device=device)

            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    except Exception as exc:  # pragma: no cover - runtime guard
        duration = time.perf_counter() - start
        return BenchmarkResult(
            name="SimpleSTBackbone",
            success=False,
            seconds=duration,
            steps=iterations,
            batch_size=batch_size,
            seq_len=seq_len,
            device=str(device),
            amp=use_amp,
            message=f"(runtime error: {exc})",
        )

    duration = time.perf_counter() - start
    return BenchmarkResult(
        name="SimpleSTBackbone",
        success=True,
        seconds=duration,
        steps=iterations,
        batch_size=batch_size,
        seq_len=seq_len,
        device=str(device),
        amp=use_amp,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def available_tasks() -> Dict[str, BenchmarkTask]:
    return {
        "simple": BenchmarkTask(
            name="SimpleSTBackbone",
            description="Tiny ST-GCN-style baseline defined in train_skeleton_gcn_simple_backbone.py",
            runner=run_simple_stgcn,
        ),
    }


def summarize_results(results: Iterable[BenchmarkResult]) -> str:
    lines = ["\nBenchmark summary:\n"]
    for res in results:
        lines.append(res.as_row())
    return "\n".join(lines)

def resolve_device(device_arg: Optional[str]) -> torch.device:
    """
    Accepts:
      - None / "" / "auto" -> "cuda" if available else "cpu"
      - "cpu"
      - "cuda"
      - "cuda:0", "cuda:1", ...

    Falls back sensibly and prints a warning to stderr on bad input.
    """
    def auto_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg in (None, "", "auto"):
        return auto_device()

    try:
        dev = torch.device(device_arg)
    except Exception as exc:
        print(
            f"[benchmark] Warning: invalid device '{device_arg}' ({exc}); falling back to auto.",
            file=sys.stderr,
        )
        return auto_device()

    if dev.type == "cuda":
        if not torch.cuda.is_available():
            print(
                f"[benchmark] Warning: CUDA requested ('{device_arg}') but CUDA is not available; using CPU.",
                file=sys.stderr,
            )
            return torch.device("cpu")

        if dev.index is not None and dev.index >= torch.cuda.device_count():
            print(
                f"[benchmark] Warning: CUDA device index {dev.index} out of range "
                f"(found {torch.cuda.device_count()} devices); falling back to cuda:0.",
                file=sys.stderr,
            )
            # If there is at least one CUDA device, cuda:0 is valid here
            return torch.device("cuda:0")

    return dev

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark skeleton models with synthetic data")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(available_tasks().keys()),
        default=list(available_tasks().keys()),
        help="Models to benchmark (default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for synthetic inputs")
    parser.add_argument("--seq-len", type=int, default=50, help="Sequence length (T) for synthetic inputs")
    parser.add_argument("--steps", type=int, default=3, help="Number of optimizer steps to run")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g. 'cpu', 'cuda', 'cuda:0', 'cuda:1' (default: auto)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False, help="Enable AMP when CUDA is available")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    tasks = available_tasks()

    results: List[BenchmarkResult] = []
    for key in args.models:
        if key not in tasks:
            print(f"Unknown model key: {key}. Choices: {list(tasks)}")
            continue
        print(f"Running {tasks[key].name}...")
        results.append(tasks[key](args))

    print(summarize_results(results))


if __name__ == "__main__":
    main()
