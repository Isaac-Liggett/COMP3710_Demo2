from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from gan import Generator, Discriminator
from dataloader import get_dataloader

from tqdm import tqdm

from datetime import datetime

import argparse

BATCH_SIZE=16
EPOCHS=100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
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
    return parser.parse_args()

LR = 1e-3
IMAGE_SIZE = 64
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE  # single-channel, flattened — see note above

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss = nn.BCELoss()

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
    args = parse_args()

    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    CHECKPOINT_DIR = Path(args.output_dir / "checkpoints")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Pytorch running on {device}")
    
    train_dataloader = get_dataloader(
        args.data_dir / "keras_png_slices_train",
        args.data_dir / "keras_png_slices_seg_train",
        image_size=256,
        num_workers=0,
        batch_size=args.batch_size,
    )

    validate_dataloader = get_dataloader(
        args.data_dir / "keras_png_slices_validate",
        args.data_dir / "keras_png_slices_seg_validate",
        image_size=256,
        num_workers=0,
        batch_size=args.batch_size,
    )

    model = UNet(n_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    val_losses = []
    checkpoint_every = 3  # epochs

    for epoch in tqdm(range(1, args.epochs + 1), desc="EPOCH"):
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
        print(f"Epoch {epoch}/{args.epochs} | avg loss: {avg_loss:.4f}, val loss: {val_loss:.4f}")

        if epoch % checkpoint_every == 0 or epoch == EPOCHS:
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                CHECKPOINT_DIR / f"unet_epoch{epoch}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pt",
            )

    plot_loss_curve(
        losses, val_losses,
        save_path=OUT_DIR / f"loss_curve_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
    )