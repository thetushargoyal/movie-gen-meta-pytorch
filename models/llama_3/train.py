from utils.Args import ModelArgs
from utils.train_utils import train
from utils.data_utils import load_dataset
from components.Transformer import Transformer
import torch

### Step 1: Input Block ###

# Using Tiny Shakespeare dataset for character-level tokenizer. Some part of the following character-level tokenizer is referenced from Andrej karpathy's GitHub (https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py) which I found is explained very well.
# Load tiny_shakespeare data file (https://github.com/tamangmilan/llama3/blob/main/tiny_shakespeare.txt)


# Define tensor token variable to be used later during model training


# Create a dataset by encoding the entire tiny_shakespeare data token_ids list using the tokenizer's encode function that we've built at the input block section
dataset, vocab, stoi, itos = load_dataset("models/llama_3/dataset/tiny_shakespeare.txt")
model = Transformer(ModelArgs).to(ModelArgs.device)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer, dataset, vocab, stoi, itos, ModelArgs)