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

class Decoder(nn.Module):
    def __init__(self, C, T, H, W):
        super(Decoder, self).__init__()
        self.H, self.W, self.T = H, W, T

        # Fully connected layer to reshape z back into 3D
        self.fc = nn.Linear(16 * (T // 8) * (H // 8) * (W // 8), 16 * (T // 8) * (H // 8) * (W // 8))

        # Temporal and spatial transpose convolutions
        self.temp_deconv3 = nn.ConvTranspose3d(16, 16, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), output_padding=(1, 0, 0))
        self.deconv3 = nn.ConvTranspose3d(16, 8, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1))

        self.temp_deconv2 = nn.ConvTranspose3d(8, 8, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), output_padding=(1, 0, 0))
        self.deconv2 = nn.ConvTranspose3d(8, 4, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1))

        self.temp_deconv1 = nn.ConvTranspose3d(4, 4, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), output_padding=(1, 0, 0))
        self.deconv1 = nn.ConvTranspose3d(4, C, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1))

    def forward(self, z):
        batch_size = z.shape[0]

        # Flatten and upsample
        z = z.view(batch_size, -1)
        z = F.relu(self.fc(z))
        z = z.view(batch_size, 16, self.T // 8, self.H // 8, self.W // 8)

        # Perform reverse convolution operations
        z = F.relu(self.temp_deconv3(z))
        z = F.relu(self.deconv3(z))

        z = F.relu(self.temp_deconv2(z))
        z = F.relu(self.deconv2(z))

        z = F.relu(self.temp_deconv1(z))
        z = self.deconv1(z)  # Last layer without activation for reconstruction

        return z


class SpatioTemporalAE(nn.Module):
    def __init__(self, C, T, H, W):
        super(SpatioTemporalAE, self).__init__()
        self.encoder = Encoder(C, T, H, W)
        self.decoder = Decoder(C, T, H, W)

    def forward(self, x):
        _, _, z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x
    
class Patchify(nn.Module):
    def __init__(self, input_dim, embed_dim, patch_size=(1, 2, 2)):
        """
        Patchify the latent video representation into tokens.
        
        Args:
        - input_dim: Number of input channels.
        - embed_dim: Dimension of the output embedding (projection dimension).
        - patch_size: Tuple (k_t, k_h, k_w) for the patch size (kernel size).
        """
        super(Patchify, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv3d(
            input_dim, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        """
        Forward pass to convert the input tensor into patchified tokens.

        Args:
        - x: Input tensor of shape (batch_size, channels, T, H, W).

        Returns:
        - tokens: Flattened sequence of patches of shape (batch_size, num_tokens, embed_dim).
        """
        x = self.projection(x)  # Shape: (batch_size, embed_dim, T', H', W')
        batch_size, embed_dim, t, h, w = x.shape
        # Flatten spatial and temporal dimensions to create a sequence
        tokens = x.view(batch_size, embed_dim, -1).permute(0, 2, 1)  # Shape: (batch_size, num_tokens, embed_dim)
        return tokens


# Input dimensions
# C, T, H, W = 3, 5*16, 64, 64  # Channels, Temporal length, Height, Width
# input_tensor = torch.randn(1, C, T, H, W)  # Example input tensor
# autoencoder = SpatioTemporalAE(C, T, H, W)
# reconstructed_x = autoencoder(input_tensor)

# print(reconstructed_x.shape)  # Output shape: torch.Size([1, 3, 80, 64, 64])