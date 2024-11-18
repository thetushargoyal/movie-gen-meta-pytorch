from models import SpatioTemporalAE
from dataset import load_sample
import torch

# Initialize the model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample = load_sample().to(device)
autoencoder = SpatioTemporalAE(3, 80, 64, 64).to(device)

# Pass through the autoencoder
reconstructed_x = autoencoder(sample)
print(reconstructed_x.shape)