from datasets import load_dataset
from utils.prepare_dataset import preprocess_video, preprocess_video_tensor
import torch

def load_sample():
    ds = load_dataset("TempoFunk/webvid-10M", streaming=True, split="train")
    sample = list(ds.take(1))
    sample = preprocess_video(sample[0])
    sample = torch.tensor(sample["processed_frames"])
    sample = torch.permute(sample, (3, 0, 1, 2))
    sample = sample.unsqueeze(0) # Add batch dimension
    sample = preprocess_video_tensor(sample)
    return sample
