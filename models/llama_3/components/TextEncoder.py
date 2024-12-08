import torch
from torch import nn

class TextEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError