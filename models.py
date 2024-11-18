import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        batch, dim = z_mean.shape
        epsilon = Normal(0, 1).sample((batch, dim)).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class Encoder(nn.Module):
    def __init__(self, C, T, H, W):
        super(Encoder, self).__init__()
        self.H, self.W, self.T = H, W, T

        # Spatial Convolutions
        self.conv1 = nn.Conv3d(C, 4, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.temp_conv1 = nn.Conv3d(4, 4, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        self.conv2 = nn.Conv3d(4, 8, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.temp_conv2 = nn.Conv3d(8, 8, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        self.conv3 = nn.Conv3d(8, 16, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.temp_conv3 = nn.Conv3d(16, 16, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        # Fully connected layers for z_mean and z_log_var
        self.flatten = nn.Flatten()

        self.fc_mean = nn.Linear(
            16 * (T // 8) * (H // 8) * (W // 8), T // 8 * H // 8 * W // 8 * 16
        )
        self.fc_log_var = nn.Linear(
            16 * (T // 8) * (H // 8) * (W // 8), T // 8 * H // 8 * W // 8 * 16
        )

        self.sampling = Sampling()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.temp_conv1(x))

        x = F.relu(self.conv2(x))
        x = F.relu(self.temp_conv2(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.temp_conv3(x))

        x = self.flatten(x)

        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)

        z = self.sampling(z_mean, z_log_var)
        z = z.view(-1, 16, self.T // 8, self.H // 8, self.W // 8)
        return z_mean, z_log_var, z

# Example Usage
C, T, H, W = 3, 5*16, 64, 64  # Channels, Height, Width, Temporal Length
input_tensor = torch.randn(1, C, T, H, W)  # 3D input (Batch, Channels, Temporal, Height, Width)

encoder = Encoder(C, T, H, W)
z_mean, z_log_var, z = encoder(input_tensor)

print(f"Shape of z_mean: {z_mean.shape}")
print(f"Shape of z_log_var: {z_log_var.shape}")
print(f"Shape of z: {z.shape}")