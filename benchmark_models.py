"""Benchmark and smoke-test the available skeleton models.

Two entry points are provided:
- CLI mode: run quick synthetic-training loops for one or more models and
  report wall-clock timings.
- GUI mode: launch a small Tkinter dashboard to pick parameters and
  models interactively.

The goal is to answer: "Can this environment run the models, and how long
might it take to scale up to bigger runs?" The benchmarks intentionally use
synthetic data so they can execute without the NTU60 dataset mounted.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import threading
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
    iterations: int
    batch_size: int
    seq_len: int
    device: str
    amp: bool
    message: str = ""

    def as_row(self) -> str:
        status = "✅" if self.success else "❌"
        return (
            f"{status} {self.name}: {self.iterations} iters, "
            f"batch={self.batch_size}, T={self.seq_len}, device={self.device}, "
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
# CTR-GCN synthetic training loop
# ---------------------------------------------------------------------------

def run_ctrgcn(args: argparse.Namespace) -> BenchmarkResult:
    start = time.perf_counter()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = args.amp and device.type == "cuda"

    ctrgcn_root = Path("/workspace/CTR-GCN")
    ms_g3d_root = Path("/workspace/MS-G3D")
    if not ctrgcn_root.exists():
        return BenchmarkResult(
            name="CTR-GCN",
            success=False,
            seconds=0.0,
            iterations=args.iterations,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=str(device),
            amp=use_amp,
            message="(skipped: /workspace/CTR-GCN not found)",
        )

    # Control sys.path so CTR-GCN takes precedence over MS-G3D in imports
    for root in (str(ctrgcn_root), str(ms_g3d_root)):
        if root in sys.path:
            sys.path.remove(root)
    sys.path.insert(0, str(ctrgcn_root))
    sys.path.append(str(ms_g3d_root))

    try:
        from model import ctrgcn  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        return BenchmarkResult(
            name="CTR-GCN",
            success=False,
            seconds=0.0,
            iterations=args.iterations,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=str(device),
            amp=use_amp,
            message=f"(import failed: {exc})",
        )

    model = ctrgcn.Model(
        num_class=60,
        num_point=25,
        num_person=2,
        graph="graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=3,
        drop_out=0.0,
        adaptive=True,
    ).to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)

    iterations = args.iterations
    batch_size = args.batch_size
    seq_len = args.seq_len

    try:
        for _ in range(iterations):
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
            name="CTR-GCN",
            success=False,
            seconds=duration,
            iterations=iterations,
            batch_size=batch_size,
            seq_len=seq_len,
            device=str(device),
            amp=use_amp,
            message=f"(runtime error: {exc})",
        )

    duration = time.perf_counter() - start
    return BenchmarkResult(
        name="CTR-GCN",
        success=True,
        seconds=duration,
        iterations=iterations,
        batch_size=batch_size,
        seq_len=seq_len,
        device=str(device),
        amp=use_amp,
    )


# ---------------------------------------------------------------------------
# Simple ST-GCN-style backbone synthetic training loop
# ---------------------------------------------------------------------------

def run_simple_stgcn(args: argparse.Namespace) -> BenchmarkResult:
    from train_skeleton_gcn_simple_backbone import SimpleSTBackbone

    start = time.perf_counter()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = args.amp and device.type == "cuda"

    model = SimpleSTBackbone(num_classes=60, T=args.seq_len).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)

    iterations = args.iterations
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
            iterations=iterations,
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
        iterations=iterations,
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
        "ctrgcn": BenchmarkTask(
            name="CTR-GCN",
            description="Synthetic AMP training loop for CTR-GCN (requires /workspace/CTR-GCN)",
            runner=run_ctrgcn,
        ),
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


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def launch_gui(args: argparse.Namespace, tasks: Dict[str, BenchmarkTask]) -> None:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk

    root = tk.Tk()
    root.title("Skeleton Action Lab Benchmarks")

    param_frame = ttk.LabelFrame(root, text="Parameters")
    param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    ttk.Label(param_frame, text="Batch size").grid(row=0, column=0, sticky="w")
    batch_var = tk.IntVar(value=args.batch_size)
    ttk.Entry(param_frame, textvariable=batch_var, width=10).grid(row=0, column=1, padx=5)

    ttk.Label(param_frame, text="Sequence length (T)").grid(row=1, column=0, sticky="w")
    seq_var = tk.IntVar(value=args.seq_len)
    ttk.Entry(param_frame, textvariable=seq_var, width=10).grid(row=1, column=1, padx=5)

    ttk.Label(param_frame, text="Iterations").grid(row=2, column=0, sticky="w")
    iter_var = tk.IntVar(value=args.iterations)
    ttk.Entry(param_frame, textvariable=iter_var, width=10).grid(row=2, column=1, padx=5)

    device_var = tk.StringVar(value=args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ttk.Label(param_frame, text="Device").grid(row=3, column=0, sticky="w")
    ttk.Combobox(param_frame, textvariable=device_var, values=["cuda", "cpu"], width=7).grid(row=3, column=1, padx=5)

    amp_var = tk.BooleanVar(value=args.amp)
    ttk.Checkbutton(param_frame, text="Enable AMP (CUDA only)", variable=amp_var).grid(row=4, column=0, columnspan=2, sticky="w")

    model_frame = ttk.LabelFrame(root, text="Models")
    model_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    model_vars: Dict[str, tk.BooleanVar] = {}
    for idx, (key, task) in enumerate(tasks.items()):
        var = tk.BooleanVar(value=True)
        model_vars[key] = var
        ttk.Checkbutton(model_frame, text=f"{task.name} – {task.description}", variable=var).grid(
            row=idx, column=0, sticky="w"
        )

    output = scrolledtext.ScrolledText(root, width=90, height=18, state=tk.DISABLED)
    output.grid(row=2, column=0, padx=10, pady=10)

    def log(message: str) -> None:
        output.configure(state=tk.NORMAL)
        output.insert(tk.END, message + "\n")
        output.see(tk.END)
        output.configure(state=tk.DISABLED)

    def run_gui_benchmarks() -> None:
        selected = [key for key, var in model_vars.items() if var.get()]
        if not selected:
            messagebox.showinfo("No models selected", "Pick at least one model to benchmark.")
            return

        run_args = argparse.Namespace(
            batch_size=batch_var.get(),
            seq_len=seq_var.get(),
            iterations=iter_var.get(),
            device=device_var.get(),
            amp=amp_var.get(),
        )

        def worker() -> None:
            log("Starting benchmarks...\n")
            results: List[BenchmarkResult] = []
            for key in selected:
                res = tasks[key](run_args)
                results.append(res)
                log(res.as_row())
            log("\n" + summarize_results(results))

        threading.Thread(target=worker, daemon=True).start()

    run_button = ttk.Button(root, text="Run Benchmarks", command=run_gui_benchmarks)
    run_button.grid(row=3, column=0, padx=10, pady=10, sticky="e")

    root.mainloop()


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
    parser.add_argument("--iterations", type=int, default=3, help="Number of optimizer steps to run")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Force a device (defaults to auto)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable AMP when CUDA is available")
    parser.add_argument("--gui", action="store_true", help="Launch the Tkinter GUI instead of CLI mode")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    tasks = available_tasks()

    if args.gui:
        launch_gui(args, tasks)
        return

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
