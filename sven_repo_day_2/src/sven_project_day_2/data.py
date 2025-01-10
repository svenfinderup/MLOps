from pathlib import Path

import typer
from torch.utils.data import Dataset

import torch
import os
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "data/corruptedmnist"

def normalize_tensor(tensor):
    """Normalize the tensor to have zero mean and unit variance."""
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std

def corrupt_mnist():
    """Return train and test datasets for corrupt MNIST."""
    # Load training data
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{DATA_PATH}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{DATA_PATH}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # Load test data
    test_images = torch.load(f"{DATA_PATH}/test_images.pt")
    test_target = torch.load(f"{DATA_PATH}/test_target.pt")

    # Normalize the images
    train_images = normalize_tensor(train_images)
    test_images = normalize_tensor(test_images)

    # Add a channel dimension (needed for CNNs)
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()

    # Convert labels to long type for classification
    train_target = train_target.long()
    test_target = test_target.long()

    # Create TensorDatasets
    train_set = TensorDataset(train_images, train_target)
    test_set = TensorDataset(test_images, test_target)

    return train_set, test_set

