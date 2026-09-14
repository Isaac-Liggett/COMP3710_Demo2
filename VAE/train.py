from pathlib import Path
import argparse

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from vae import VAE
from dataloader import get_dataloader

BATCH_SIZE = 3
EPOCHS = 50
LR = 1e-3
IMAGE_SIZE = 256
IN_CHANNELS = 1
HIDDEN_DIMS = [32, 64, 128, 256, 512]
LATENT_DIM = 256
OUTPUT_ACTIVATION = "sigmoid"
LIKELIHOOD = "gaussian"
BETA = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./keras_png_slices_data"),
        help="Path to the keras_png_slices_data folder",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Path to the checkpoints and graphs parent folder",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    return parser.parse_args()


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


def plot_loss_curve(recon_losses, kld_losses, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(recon_losses) + 1)

    ax[0].plot(epochs, recon_losses, marker="o")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Avg reconstruction loss (per sample)")
    ax[0].set_title("Reconstruction Loss")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(epochs, kld_losses, marker="o", color="darkorange")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Avg KL divergence (per sample)")
    ax[1].set_title("KL Divergence")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Returns (total_loss, recon_loss, kld) -- summed over the batch (not
    averaged), consistent with reduction="sum" used elsewhere in this
    project. Log recon_loss and kld separately during training to check
    for posterior collapse (kld -> 0) or an unregularized latent space
    (kld staying large while recon dominates).
    """
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction="sum")
 
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld


if __name__ == "__main__":
    print(f"Pytorch running on {device}")

    args = parse_args()

    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = OUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    loss_curve_path = args.output_dir / "loss_curve.png"

    dataloader = get_dataloader(
        str(args.data_dir / "keras_png_slices_train"),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=0
    )

    model = VAE(
        in_channels=IN_CHANNELS,
        latent_dim=args.latent_dim,
        hidden_dims=HIDDEN_DIMS,
        image_size=args.image_size,
        output_activation=OUTPUT_ACTIVATION,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    recon_losses = []
    kld_losses = []

    for epoch in tqdm(range(1, args.epochs + 1), desc="EPOCH"):
        model.train()
        epoch_recon = 0.0
        epoch_kld = 0.0
        for data in tqdm(dataloader, desc="BATCH", leave=False):
            data = data.to(device)  # expects [N, C, H, W]; no flattening for conv model
            optimizer.zero_grad()
            recon_batch, _, mu, logvar = model(data)
            loss, recon_loss, kld = loss_function(
                recon_batch, data, mu, logvar, beta=args.beta
            )
            loss.backward()
            epoch_recon += recon_loss.item()
            epoch_kld += kld.item()
            optimizer.step()

        n = len(dataloader.dataset)
        avg_recon = epoch_recon / n
        avg_kld = epoch_kld / n
        recon_losses.append(avg_recon)
        kld_losses.append(avg_kld)
        print(f"Epoch {epoch}/{args.epochs} | recon: {avg_recon:.4f}  kld: {avg_kld:.4f}")

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(
                model, optimizer, epoch, avg_recon + args.beta * avg_kld,
                checkpoint_dir / f"vae_epoch{epoch}.pt",
            )

    plot_loss_curve(recon_losses, kld_losses, loss_curve_path)