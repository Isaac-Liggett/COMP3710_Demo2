import argparse
import random
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from vae import VAE
from dataloader import get_dataloader

# These MUST match the architecture used when the checkpoint was trained --
# loading a checkpoint into a differently-shaped model will fail (parameter
# shape mismatch) or, worse, load silently wrong if shapes happen to
# coincide. If you've changed these in train.py since a given checkpoint was
# saved, use the values that were active for THAT checkpoint, not today's.
IMAGE_SIZE = 256
IN_CHANNELS = 1
HIDDEN_DIMS = [32, 64, 128, 256, 512]
LATENT_DIM = 256
OUTPUT_ACTIVATION = "sigmoid"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                         help="Path to a .pt checkpoint saved by train.py")
    parser.add_argument("--data-dir", type=Path, default=Path("./keras_png_slices_data"),
                         help="Path to keras_png_slices_data (used to pick a random real image)")
    parser.add_argument("--split", type=str, default="keras_png_slices_validate",
                         help="Which image subfolder to draw the random sample from")
    parser.add_argument("--num-generated", type=int, default=8,
                         help="Number of images to sample from the prior N(0, I)")
    parser.add_argument("--output", type=Path, default=Path("./vae_samples.png"))
    parser.add_argument("--seed", type=int, default=None,
                         help="Optional seed for reproducible image choice / sampling")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the model with the SAME architecture as training, then load weights.
    model = VAE(
        in_channels=IN_CHANNELS,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        image_size=IMAGE_SIZE,
        output_activation=OUTPUT_ACTIVATION,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
          f"(loss {checkpoint.get('loss', float('nan')):.4f})")

    # --- Pick one real image at random and reconstruct it ---
    dataloader = get_dataloader(
        str(args.data_dir / args.split),
        image_size=IMAGE_SIZE,
        batch_size=1,
        num_workers=0,
        shuffle=True,
    )
    real_img = next(iter(dataloader)).to(device)  # [1, C, H, W]

    with torch.no_grad():
        recon_img, _, mu, logvar = model(real_img)
        # Note: this uses the STOCHASTIC reparameterized z (mu + eps*std), not
        # mu directly, so re-running this script on the same image without
        # --seed will give a slightly different reconstruction each time.
        # Use model.decode(mu) instead of model(real_img) if you want the
        # deterministic "best guess" reconstruction with no sampling noise.

        # --- Sample fresh images directly from the prior N(0, I) ---
        generated = model.sample(args.num_generated, device)

    # --- Plot: original | reconstruction | generated samples ---
    n_cols = 2 + args.num_generated
    fig, axes = plt.subplots(1, n_cols, figsize=(2 * n_cols, 2.5))

    axes[0].imshow(real_img[0, 0].cpu(), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon_img[0, 0].cpu(), cmap="gray")
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    for i in range(args.num_generated):
        axes[2 + i].imshow(generated[i, 0].cpu(), cmap="gray")
        axes[2 + i].set_title(f"Sample {i+1}")
        axes[2 + i].axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()