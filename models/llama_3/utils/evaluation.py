import torch
from .Args import ModelArgs 
from .data_utils import get_dataset_batch
import numpy as np

# Define a evaluate loss function to calculate and store training and validation loss for logging and plotting
@torch.no_grad()
def evaluate_loss(model, dataset, vocab, stoi, itos, args: ModelArgs):
  out = {}
  model.eval()

  for split in ["train", "val"]:
    losses = []
    for _ in range(10):
      xb, yb = get_dataset_batch(dataset, 'train', vocab, stoi, itos, args)
      _, loss = model(x=xb, targets=yb)
      losses.append(loss.item())
    out[split] = np.mean(losses)

  model.train()
  return out