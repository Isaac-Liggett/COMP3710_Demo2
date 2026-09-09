import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    Implementation of a VAE taken from
    https://medium.com/@mikelgda/implementing-a-variational-autoencoder-in-pytorch-ddc0bb5ea1e7
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.input_dims = input_dim
        self.hidden_dims = hidden_dim
        self.latent_dims = latent_dim

        # Initialize the encoder layers
        # first linear
        self.linear_encode = nn.Linear(input_dim, hidden_dim)
        # Mean of latent space
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        # Log variance of latent space  
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)  

        # Decoder layers
        # first linear
        self.linear_decode = nn.Linear(latent_dim, hidden_dim)
        # map to likelihood parameters
        self.linear_likelihood = nn.Linear(
            hidden_dim, input_dim
        )

    def encode(self, x):
        h = nn.functional.relu(self.linear_encode(x))
        return self.linear_mu(h), self.linear_logvar(h)

    def decode(self, z):
        h = nn.functional.relu(self.linear_decode(z))  
        return self.linear_likelihood(h)
    
    def sample(self, mu, logvar):
        """Sample from Gaussian posterior of z with reparametrization"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x) # get posterior parameters
        z = self.sample(mu, logvar) # sample with reparametrization
        recon_x = self.decode(z) # get likelihood parameters
        return recon_x, mu, logvar

    def loss_function(recon_x, x, mu, logvar, likelihood="gaussian", beta=1.0):
        """Computes the VAE loss."""
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
        
        # KL divergence between the posterior and the prior (both Gaussian)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + beta * KLD


    
