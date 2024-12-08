from .args import ModelArgs
from .data_utils import get_dataset_batch
from .evaluation import evaluate_loss
import time
import pandas as pd

# Define a training function to perform model training
def train(model, optimizer, dataset, vocab, stoi, itos, args: ModelArgs):
    epochs = args.epochs
    log_interval = args.log_interval
    device = args.device
    losses = []
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()

        xs, ys = get_dataset_batch(dataset, 'train', vocab, stoi, itos, args)
        xs = xs.to(device)
        ys = ys.to(device)
        logits, loss = model(x=xs, targets=ys)
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model, dataset, vocab, stoi, itos, args)
            losses += [x]
            print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f}")
            start_time = time.time()

    # Print the final validation loss
    print("validation loss: ", losses[-1]['val'])
    # Display the interval losses in plot
    return pd.DataFrame(losses).plot()