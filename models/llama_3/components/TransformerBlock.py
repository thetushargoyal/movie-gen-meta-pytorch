from torch import nn
from utils.args import ModelArgs
from .RMS_Norm import RMSNorm
from .MultiHeadAttention import Attention
from .FeedForward import FeedForward

## Step2f: The Decoder Block. The class name is assigned as TransformerBlock to match the name of Meta llama 3 code base.

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # Self-attention normalization
        self.attention_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        # Self-attention module
        self.self_attention = Attention(args)
        # Cross-attention normalization
        self.cross_attention_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        # Cross-attention module
        self.cross_attention = Attention(args)
        # Feedforward normalization
        self.ff_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        # Feedforward module
        self.feedforward = FeedForward(args, args.dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)

    def forward(self, x, start_pos, inference, prompt_embedding):
        """
        Args:
            x: [batch_size, seq_len, dim] - Input embeddings for self-attention.
            start_pos: Current token position during inference mode.
            inference: Whether the model is in inference mode (bool).
            prompt_embedding: [batch_size, prompt_len, dim] - Text prompt embeddings for cross-attention.
        """
        # Self-attention
        x_norm = self.attention_norm(x)
        h = x + self.self_attention(x_norm, x_norm, x_norm, start_pos, inference)

        # Cross-attention using text prompt embedding
        h_norm = self.cross_attention_norm(h)
        h = h + self.cross_attention(h_norm, prompt_embedding, prompt_embedding, start_pos, inference)

        # Feedforward
        h_ff_norm = self.ff_norm(h)
        out = h + self.feedforward(h_ff_norm)

        return out
