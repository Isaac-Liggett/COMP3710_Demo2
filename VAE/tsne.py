from sklearn.manifold import TSNE
import torch

from vae import VAE
from dataloader import get_dataloader

import matplotlib.pyplot as plt

IN_CHANNELS = 1
HIDDEN_DIMS = [32, 64, 128, 256, 512]
LATENT_DIM = 256
OUTPUT_ACTIVATION = "sigmoid"
IMAGE_SIZE = 256

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        model_path = "C:/Users/isaac/OneDrive/Documents/University/Y3S2/COMP3710/demo_data/vae_data/checkpoints/vae_epoch50.pt"
        model = VAE(
            in_channels=IN_CHANNELS,
            hidden_dims=HIDDEN_DIMS,
            latent_dim=LATENT_DIM,
            image_size=IMAGE_SIZE,
            output_activation=OUTPUT_ACTIVATION,
        )
        state_dict = torch.load(model_path)["model_state_dict"]
        model.load_state_dict(state_dict)
        model.to(device)

        model.eval()

        # Dataset
        dataloader = get_dataloader("./keras_png_slices_data/keras_png_slices_seg_train", batch_size=200, image_size=IMAGE_SIZE)

        batch = next(iter(dataloader)).to(device)
        _, _, mu, _ = model.forward(batch)

        # Visualization
        tsne = TSNE(n_components=2).fit_transform(mu.cpu())

        plt.title("TSNE Visualization of OASIS dataset 200 sample batch")
        plt.xlabel("tsne dim 1")
        plt.ylabel("tsne dim 2")
        plt.scatter(tsne[:, 0], tsne[:, 1])
        plt.show()