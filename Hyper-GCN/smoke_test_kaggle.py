"""
Quick Hyper-GCN smoke test on the Kaggle NTU60 NPZ.

Loads a tiny subset (debug=True) of the Kaggle file, runs a single forward/backward
step, and reports loss/accuracy. Intended to verify imports, data loading, and
model wiring inside the Docker image.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from feeders.feeder_ntu import Feeder
from model.hypergcn_base import Model


def main():
    parser = argparse.ArgumentParser(description="Hyper-GCN Kaggle smoke test")
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/NTU60_CS.npz",
        help="Path to Kaggle NTU60 NPZ with x_train/x_test/y_train/y_test",
    )
    parser.add_argument("--device", type=str, default="cpu", help="torch device, e.g., cuda:0")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size for the smoke step")
    parser.add_argument("--window-size", type=int, default=48, help="temporal length after resize")
    parser.add_argument("--hyper-joints", type=int, default=3, help="virtual joints for Hyper-GCN")
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="run on small random tensors instead of loading the Kaggle NPZ (useful when the file is unavailable)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        graph="graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "virtual_ensemble"},
        hyper_joints=args.hyper_joints,
    ).to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    if args.use_synthetic or not Path(args.data_path).exists():
        if not args.use_synthetic:
            print(f"Warning: {args.data_path} not found, running synthetic smoke test instead.")
        data = torch.randn(args.batch_size, 3, args.window_size, 25, 2, device=device, dtype=torch.float32)
        labels = torch.randint(0, 60, (args.batch_size,), device=device, dtype=torch.long)
    else:
        dataset = Feeder(
            data_path=args.data_path,
            split="train",
            debug=True,
            window_size=args.window_size,
            p_interval=[0.5, 0.75],
            frame_sample="resize",
            random_rot=False,
            vel=False,
            bone=False,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        data, labels, _ = next(iter(loader))
        data = data.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

    optimizer.zero_grad()
    logits, _ = model(data)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

    print(
        f"Smoke test ok - batch_size={args.batch_size}, window={args.window_size}, "
        f"loss={loss.item():.4f}, acc={acc*100:.1f}%"
    )


if __name__ == "__main__":
    main()
