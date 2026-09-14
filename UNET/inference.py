import random

import torch
import matplotlib.pyplot as plt

from unet import UNet
from dataloader import OasisSegmentationDataset

PATH = "C:/Users/isaac/OneDrive/Documents/University/Y3S2/COMP3710/demo_data/data/checkpoints/unet_epoch100_20260914-133830.pt"
IMAGE_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print(f"Pytorch running on {device}")

    dataset = OasisSegmentationDataset(
        "./keras_png_slices_data/keras_png_slices_test",
        "./keras_png_slices_data/keras_png_slices_seg_test",
        image_size=IMAGE_SIZE,
    )

    idx = random.randrange(len(dataset))
    data, label = dataset[idx]  # data: (1, H, W), label: (4, H, W) one-hot

    model = UNet(n_classes=4).to(device)
    checkpoint = torch.load(PATH, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        input_batch = data.unsqueeze(0).to(device)  # (1, 1, H, W)
        pred = model(input_batch)
        pred_classes = pred.argmax(dim=1).squeeze(0).cpu()  # (H, W)

    input_img = data.squeeze(0).cpu()  # (H, W)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(input_img, cmap="gray")
    axes[0].set_title(f"Input (test idx {idx})")
    axes[0].axis("off")

    axes[1].imshow(pred_classes, cmap="viridis", vmin=0, vmax=3)
    axes[1].set_title("Predicted segmentation")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()