from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from unet import UNet
from unet import DiceLoss
from dataloader import get_dataloader

from tqdm import tqdm

from datetime import datetime

import argparse

BATCH_SIZE = 16
PATH = "C:/Users/isaac/OneDrive/Documents/University/Y3S2/COMP3710/demo_data/data/checkpoints/unet_epoch10_20260914-103236.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiceCELoss(nn.Module):
    """
    Combined Dice + CrossEntropy loss.

    DiceLoss (from unet.py) expects one-hot float targets (B, C, H, W).
    CrossEntropyLoss expects integer class-index targets (B, H, W).
    This wrapper assumes `targets` arrives one-hot (matching DiceLoss's
    existing contract) and derives the index targets from it via argmax.
    """
    def __init__(self, dice_weight=1.0, ce_weight=1.0, smooth=1e-5, class_weights=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss(smooth=smooth)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets_onehot):
        dice_loss = self.dice(logits, targets_onehot)
        targets_idx = targets_onehot.argmax(dim=1)  # (B, H, W)
        ce_loss = self.ce(logits, targets_idx)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss

# Inverse-frequency class weights from measured pixel proportions:
# class 0 (background): 72.2309%, class 1: 6.8700%, class 2: 10.8150%, class 3: 10.0842%
# Normalized so mean weight = 1, keeping overall CE loss scale comparable to the unweighted case.
_class_pixel_freqs = torch.tensor([0.722309, 0.068700, 0.108150, 0.100842])
_class_weights = (1.0 / _class_pixel_freqs)
_class_weights = _class_weights / _class_weights.mean()

loss_fn = DiceCELoss(class_weights=_class_weights.to(device))

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

if __name__ == "__main__":
    print(f"Pytorch running on {device}")

    test_dataloader = get_dataloader(
        "./keras_png_slices_data/keras_png_slices_test",
        "./keras_png_slices_data/keras_png_slices_seg_test",
        image_size=256,
        num_workers=0,
        batch_size=BATCH_SIZE,
    )

    model = UNet(n_classes=4).to(device)
    checkpoint = torch.load(PATH, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for data, label in tqdm(test_dataloader, desc="BATCH", leave=False):
            data = data.to(device)
            label = label.to(device)

            pred = model(data)

            loss = loss_fn(pred, label)
            epoch_loss += loss.item()

            # Overall pixel accuracy. Note: with ~72% background pixels, a
            # trivial all-background prediction would already score ~0.72
            # here — treat this as a sanity check, not a segmentation
            # quality metric. Per-class IoU/Dice would be more informative.
            pred_classes = pred.argmax(dim=1)      # (B, H, W)
            target_classes = label.argmax(dim=1)   # (B, H, W)
            correct_pixels += (pred_classes == target_classes).sum().item()
            total_pixels += target_classes.numel()

        avg_loss = epoch_loss / len(test_dataloader)
        pixel_accuracy = correct_pixels / total_pixels

    print(f"avg_loss: {avg_loss}")
    print(f"pixel_accuracy: {pixel_accuracy:.4f}")