from pathlib import Path

import torch
import matplotlib.pyplot as plt

from vae import VAE
from dataloader import get_dataloader

from tqdm import tqdm

BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
IMAGE_SIZE = 64
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE  # single-channel, flattened — see note above
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss = reconstruction + beta * KL divergence."""
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld

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

def plot_loss_curve(losses, save_path="loss_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss (per sample)")
    plt.title("VAE Training Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

# Training Loop
if __name__ == "__main__":
    print(f"Pytorch running on {device}")
    
    dataloader = get_dataloader(
        "./keras_png_slices_data/keras_png_slices_seg_train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    model = VAE(INPUT_DIM, 256, 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    checkpoint_every = 5  # epochs

    for epoch in tqdm(range(1, EPOCHS + 1), desc="EPOCH"):
        model.train()
        epoch_loss = 0.0
        for data in tqdm(dataloader, desc="BATCH", leave=False):
            data = data.to(device).view(-1, model.input_dims)  # flatten
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        avg_loss = epoch_loss / len(dataloader.dataset)

        losses.append(avg_loss)
        print(f"Epoch {epoch}/{EPOCHS}  avg loss: {avg_loss:.4f}")

        if epoch % checkpoint_every == 0 or epoch == EPOCHS:
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                CHECKPOINT_DIR / f"vae_epoch{epoch}.pt",
            )

    plot_loss_curve(losses)