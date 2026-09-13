from pathlib import Path

import torch
import matplotlib.pyplot as plt

from unet import UNet
from unet import DiceLoss
from dataloader import get_dataloader

from tqdm import tqdm

from datetime import datetime

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
IMAGE_SIZE = 64
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE  # single-channel, flattened — see note above
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dice_loss = DiceLoss()

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

def plot_loss_curve(losses, val_losses, save_path="loss_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", label="Training")
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker="s", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("UNet Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            loss = loss_fn(pred, label)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training Loop
if __name__ == "__main__":
    print(f"Pytorch running on {device}")
    
    train_dataloader = get_dataloader(
        "./keras_png_slices_data/keras_png_slices_train",
        "./keras_png_slices_data/keras_png_slices_seg_train",
        image_size=256,
        num_workers=0,
        batch_size=16
    )

    validate_dataloader = get_dataloader(
        "./keras_png_slices_data/keras_png_slices_validate",
        "./keras_png_slices_data/keras_png_slices_seg_validate",
        image_size=256,
        num_workers=0,
        batch_size=16
    )

    model = UNet(n_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    val_losses = []
    checkpoint_every = 3  # epochs

    for epoch in tqdm(range(1, EPOCHS + 1), desc="EPOCH"):
        model.train()
        epoch_loss = 0.0
        for data, label in tqdm(train_dataloader, desc="BATCH", leave=False):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(data)

            loss = dice_loss(pred, label)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        avg_loss = epoch_loss / len(train_dataloader.dataset)

        # Run validation pass
        val_loss = validate(model, validate_dataloader, dice_loss, device)
        val_losses.append(val_loss)

        losses.append(avg_loss)
        print(f"Epoch {epoch}/{EPOCHS} | avg loss: {avg_loss:.4f}, val loss: {val_loss:.4f}")

        if epoch % checkpoint_every == 0 or epoch == EPOCHS:
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                CHECKPOINT_DIR / f"unet_epoch{epoch}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pt",
            )

    plot_loss_curve(losses, val_losses, save_path=f"./loss_curve_{datetime.now().strftime("%Y%m%d-%H%M%S")}.png")