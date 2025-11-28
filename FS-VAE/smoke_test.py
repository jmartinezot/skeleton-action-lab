import argparse

import torch

from model import Decoder, Encoder, KL_divergence, reparameterize


def main():
    parser = argparse.ArgumentParser(description="FS-VAE smoke test")
    parser.add_argument("--device", default="cuda:0", help="device to use, e.g., cuda:0 or cpu")
    parser.add_argument("--vis-dim", type=int, default=256, help="visual embedding dimension")
    parser.add_argument("--text-dim", type=int, default=1024, help="text embedding dimension")
    parser.add_argument("--latent-dim", type=int, default=32, help="latent dimension")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    print(f"Running smoke test on {device}")

    # Build tiny encoder/decoder pairs
    seq_enc = Encoder([args.vis_dim, args.latent_dim]).to(device)
    seq_dec = Decoder([args.latent_dim, args.vis_dim]).to(device)
    txt_enc = Encoder([args.text_dim, args.latent_dim]).to(device)
    txt_dec = Decoder([args.latent_dim, args.text_dim]).to(device)

    # Fake inputs
    vis = torch.randn(4, args.vis_dim, device=device)
    txt = torch.randn(4, args.text_dim, device=device)

    # Forward/backward passes
    smu, slv = seq_enc(vis)
    sz = reparameterize(smu, slv)
    vis_out = seq_dec(sz)

    tmu, tlv = txt_enc(txt)
    tz = reparameterize(tmu, tlv)
    txt_out = txt_dec(tz)

    # Simple losses
    recon_loss = torch.nn.functional.mse_loss(vis_out, vis) + torch.nn.functional.mse_loss(txt_out, txt)
    kld_loss = KL_divergence(smu, slv) + KL_divergence(tmu, tlv)
    loss = recon_loss + kld_loss
    loss.backward()

    print(f"Smoke test ok. Loss={loss.item():.4f}, recon={recon_loss.item():.4f}, kld={kld_loss.item():.4f}")


if __name__ == "__main__":
    main()
