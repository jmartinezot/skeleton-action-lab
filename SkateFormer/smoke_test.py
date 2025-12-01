import argparse

import torch

from model.SkateFormer import SkateFormer_


def main():
    parser = argparse.ArgumentParser(description="SkateFormer smoke test")
    parser.add_argument("--device", default="cpu", help="torch device, e.g., cpu or cuda:0")
    parser.add_argument("--batch-size", type=int, default=2, help="batch size for the synthetic step")
    parser.add_argument("--frames", type=int, default=32, help="temporal length for the synthetic tensor")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    print(f"Running SkateFormer smoke test on {device}")

    model = SkateFormer_(num_frames=args.frames, num_points=25, num_people=2, num_classes=60).to(device)
    model.train()

    data = torch.randn(args.batch_size, 3, args.frames, 25, 2, device=device)
    labels = torch.randint(0, 60, (args.batch_size,), device=device)
    time_indices = torch.arange(args.frames, device=device).expand(args.batch_size, -1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    logits = model(data, time_indices)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    print(f"Smoke test ok - frames={args.frames}, loss={loss.item():.4f}, acc={acc*100:.1f}%")


if __name__ == "__main__":
    main()
