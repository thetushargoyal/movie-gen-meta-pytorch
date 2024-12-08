# Import necessary libraries
import torch
from torch import nn
from utils.Args import ModelArgs

class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    global device
    device = ModelArgs.device
    self.eps = eps
    # Scaling parameter gamma, initialized with one and the no of parameters is equal to the size of dim
    self.weight = nn.Parameter(torch.ones(dim).to(device))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(device)

  def forward(self, x):
    #Shape: x[bs,seq,dim]
    output = self._norm(x.float()).type_as(x)

    #Shape: x[bs,seq,dim] -> x_norm[bs,seq,dim]
    return output * self.weight