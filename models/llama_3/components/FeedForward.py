import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
from utils.args import ModelArgs

class FeedForward(nn.Module):
  def __init__(self, args: ModelArgs, dim:int, hidden_dim:int, multiple_of:int, ffn_dim_multiplier: Optional[float]):
    super().__init__()
    # Models embedding dimension
    self.dim = dim

    # We must use the hidden dimensions calculation shared by Meta which is the ideal one for this model
    # Hidden dimension are calculated such that it is a multiple of 256.
    hidden_dim = int(2 * hidden_dim/3)
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    # define hiddne layers weights
    self.w1 = nn.Linear(self.dim, hidden_dim, bias=False, device=args.device)
    self.w2 = nn.Linear(hidden_dim, self.dim, bias=False, device=args.device)
    self.w3 = nn.Linear(self.dim, hidden_dim, bias=False, device=args.device)

  def forward(self, x):
    # Shape: [bsz,seq_len,dim]
    return self.w2(F.silu(self.w1(x)) * self.w3(x))