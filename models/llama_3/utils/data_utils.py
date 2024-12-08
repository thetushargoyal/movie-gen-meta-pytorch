import torch
from .args import ModelArgs
import numpy as np

# Tokenizers encode function: take a string, output a list of integers
def encode(s, stoi):
  return [stoi[ch] for ch in s]

# Tokenizers decode function: take a list of integers, output a string
def decode(l, itos):
  return ''.join(itos[i] for i in l)

def load_dataset(path):
    # Load tiny_shakespeare data file.
    with open(path, 'r') as f:
        data = f.read()

    # Prepare vocabulary by taking all unique characters from the data
    vocab = sorted(list(set(data)))

    # Add special tokens to the vocabulary (only once)
    special_tokens = ['<|begin_of_text|>', '<|end_of_text|>', '<|pad_id|>']
    vocab.extend(special_tokens)

    # Create mappings
    itos = {i: ch for i, ch in enumerate(vocab)}
    stoi = {ch: i for i, ch in enumerate(vocab)}

    # Update ModelArgs.vocab_size
    ModelArgs.vocab_size = len(vocab)

    # Encode the dataset
    dataset = torch.tensor(encode(data, stoi), dtype=torch.int).to(ModelArgs.device)

    # print(f"Dataset shape: {dataset.shape}, Vocab size: {ModelArgs.vocab_size}")
    return dataset, vocab, stoi, itos

# Define function to generate batches from the given dataset
def get_dataset_batch(data, split, vocab, stoi, itos, args: ModelArgs):
    seq_len = args.max_seq_len
    batch_size = args.max_batch_size
    device = args.device

    # Split data into train, val, and test
    train = data[:int(0.8 * len(data))]
    val = data[int(0.8 * len(data)): int(0.9 * len(data))]
    test = data[int(0.9 * len(data)):]

    batch_data = train
    if split == "val":
        batch_data = val
    elif split == "test":
        batch_data = test

    # Get token indices for special tokens
    token_bos = torch.tensor([stoi['<|begin_of_text|>']], dtype=torch.int, device=device)
    token_eos = torch.tensor([stoi['<|end_of_text|>']], dtype=torch.int, device=device)

    # Generate random indices for the batch
    ix = torch.randint(0, len(batch_data) - seq_len - 1, (batch_size,)).to(device)

    # Create x and y batches
    x = torch.stack([torch.cat([token_bos, batch_data[i:i + seq_len - 1]]) for i in ix]).long().to(device)
    y = torch.stack([torch.cat([batch_data[i + 1:i + seq_len], token_eos]) for i in ix]).long().to(device)

    return x, y
