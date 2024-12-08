from torch import nn
from torch.nn import functional as F
from utils.args import ModelArgs
from .TransformerBlock import TransformerBlock
from .RMS_Norm import RMSNorm

## Step3: The Output Block
# This is the Llama 3 model. Again, the class name is maintained as Transformer to match with Meta Llama 3 model.
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        # Set all ModelArgs in params variable
        self.params = params

        # Initialize embedding class from the input block
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # Initialize the decoder block and store in a ModuleList
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(args=params))

        # Initialize RMSNorm for the output block
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # Initialize linear layer at the output block
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    def forward(self, x, prompt_embedding, start_pos=0, targets=None):
        """
        Args:
            x: [batch_size, seq_len] - Input token IDs.
            prompt_embedding: [batch_size, prompt_len, dim] - Text prompt embeddings for cross-attention.
            start_pos: Token position for inference.
            targets: [batch_size, seq_len] or None - Target token IDs for loss computation.
        """
        # Token embeddings: x[batch_size, seq_len] -> h[batch_size, seq_len, dim]
        h = self.tok_embeddings(x)

        # Determine if we are in inference mode
        inference = targets is None

        # Pass embeddings through all decoder blocks
        for layer in self.layers:
            h = layer(h, start_pos, inference, prompt_embedding)

        # Normalize the output embeddings
        h = self.norm(h)

        # Map embeddings to logits: h[batch_size, seq_len, dim] -> logits[batch_size, seq_len, vocab_size]
        logits = self.output(h).float()

        # Compute loss if in training mode
        loss = None
        if not inference:
            loss = F.cross_entropy(
                logits.view(-1, self.params.vocab_size),
                targets.view(-1)
            )

        return logits, loss