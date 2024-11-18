import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

# define a class for sampling
# this class will be used in the encoder for sampling in the latent space
class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        # get the shape of the tensor for the mean and log variance
        batch, dim = z_mean.shape
        # generate a normal random tensor (epsilon) with the same shape as z_mean
        # this tensor will be used for reparameterization trick
        epsilon = Normal(0, 1).sample((batch, dim)).to(z_mean.device)
        # apply the reparameterization trick to generate the samples in the
        # latent space
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class Encoder(nn.Module):
    def __init__(self, C, H, W):
        super(Encoder, self).__init__()
        self.H, self.W = H, W 
        self.conv1 = nn.Conv2d(C, 4, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, stride=2, padding=1)

        self.flatten = nn.Flatten()

        self.fc_mean = nn.Linear(
            16 * (H // 8) * (W // 8), H//8*W//8*16
        )
        self.fc_log_var = nn.Linear(
            16 * (H // 8) * (W // 8), H//8*W//8*16
        )

        self.sampling = Sampling()
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.flatten(x)

        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)

        z = self.sampling(z_mean, z_log_var)
        z = z.view(-1, 16, self.H // 8, self.W // 8)
        return z_mean, z_log_var, z
    
# Define input dimensions and channels
C, H, W = 3, 64, 64  # Example: 3 channels (RGB), 64x64 image

# Create a random tensor as input (batch size of 8)
input_tensor = torch.randn(1, C, H, W)

# Instantiate the Encoder
encoder = Encoder(C, H, W)

# Send the input tensor through the encoder
z_mean, z_log_var, z = encoder(input_tensor)

# Print the shapes of the outputs
print(f"Shape of z_mean: {z_mean.shape}")  # Latent mean
print(f"Shape of z_log_var: {z_log_var.shape}")  # Latent log variance
print(f"Shape of z: {z.shape}")  # Sampled latent space~