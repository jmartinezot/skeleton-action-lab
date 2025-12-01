import argparse

import torch

from train_npz_mlp import SimpleMLP


def main():
    parser = argparse.ArgumentParser(description="Baselines smoke test")
    parser.add_argument("--device", default="cpu", help="torch device, e.g., cpu or cuda:0")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size for the synthetic step")
    parser.add_argument("--seq-len", type=int, default=16, help="temporal length for the synthetic tensor")
    parser.add_argument(
        "--feature-dim", type=int, default=150, help="feature dimension per timestep (matches NTU60 joint layout)"
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    print(f"Running baselines smoke test on {device}")

    model = SimpleMLP(args.seq_len * args.feature_dim, num_classes=60).to(device)
    model.train()

    data = torch.randn(args.batch_size, args.seq_len, args.feature_dim, device=device)
    labels = torch.randint(0, 60, (args.batch_size,), device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    logits = model(data)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    print(f"Smoke test ok - loss={loss.item():.4f}, acc={acc*100:.1f}%")


if __name__ == "__main__":
    main()
