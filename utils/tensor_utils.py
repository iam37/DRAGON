import torch
import numpy as np

def load_tensor(filename, tensors_path):
    """Load a Torch tensor from disk."""
    return torch.load(tensors_path / (filename + ".pt"))
