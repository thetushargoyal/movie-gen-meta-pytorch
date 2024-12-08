from dataclasses import dataclass
from typing import Optional
import torch

# Step2: The Decoder Block
# Note: Since the Llama 3 model is developed by Meta, so to be in sync with their codebase and for future compatibility,
# I will use most of the code from Meta GitHub with some necessary changes required to achieve our goal.

# Define parameters dataclass: we'll use these parameters during model building, training and inference.
# Note: Since we want to see the results of training and inferencing faster rather than focusing on high accuracy, we're taking lower values for most of the parameters which are set higher in the Llama 3 model.

@dataclass
class ModelArgs:
    dim: int = 512              # embedding dimension
    n_layers: int = 8           # number of model decoder blocks
    n_heads: int = 8            # number of heads for queries embedding
    n_kv_heads: int = 4         # number of heads for keys and values embedding
    vocab_size: Optional[int] = None # Length of vocabulary
    multiple_of: int = 256        # Require to calculate dim of feedfoward network
    ffn_dim_multiplier: Optional[float] = None  # Require to calculate dim of feedfoward network
    norm_eps: float = 1e-5                       # Default Epsilon value set for the RMSNorm calculation
    rope_theta: float = 10000.0   # Default theta value for the RePE calculation

    max_batch_size: int = 10      # Max batch size
    max_seq_len: int = 256         # Max sequence length

    epochs: int = 2500             # Total number of training iteration
    log_interval: int = 10        # Number of interval to print the logs and loss values
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'   # Assign device to cuda or cpu based on availability

