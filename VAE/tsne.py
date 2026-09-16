from sklearn.manifold import TSNE
import torch

from vae import VAE
from dataloader import get_dataloader

import matplotlib.pyplot as plt

if __name__ == "__main__":
    with torch.no_grad():
        model_path = "C:/Users/isaac/OneDrive/Documents/University/Y3S2/COMP3710/demo_data/vae_data/checkpoints/vae_epoch50.pt"
        model = VAE(64*64, 256)
        state_dict = torch.load(model_path)["model_state_dict"]
        model.load_state_dict(state_dict)

        model.eval()

        # Dataset
        dataloader = get_dataloader("./keras_png_slices_data/keras_png_slices_seg_train", batch_size=200, image_size=64)

        batch = next(iter(dataloader)).view(-1, model.input_dims)
        pred = model.forward(batch)[1]

        # Visualization
        tsne = TSNE(n_components=2).fit_transform(pred.detach().cpu().numpy())

        plt.scatter(tsne[:, 0], tsne[:, 1])
        plt.show()