from models import SpatioTemporalAE, Patchify
from dataset import load_sample
import torch

# Initialize the model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample = load_sample().to(device)

Patchifer = Patchify(3, 100).to(device)
input_tokens = Patchifer(sample)

# autoencoder = SpatioTemporalAE(3, 80, 64, 64).to(device)
# # Pass through the autoencoder
# reconstructed_x = autoencoder(sample)
# print(reconstructed_x.shape)
print(sample.shape)  # Expected output: torch.Size([1, 3, 80, 64, 64])
print(input_tokens.shape)  # Expected output: torch.Size([1, 200, 100])