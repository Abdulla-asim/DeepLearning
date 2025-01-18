"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    """ Creates trainig and testing DataLoaders

    Takes in a training direcotry and testing directory path and
    turns them into PyTorch Datasets and them into PyTorch Dataloaders.
    
    Args: 
        train_dir:  Path to trainig directory.
        test_dir:   Path to testing directory.
        transfrom:  torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoader
        num_workers: An integer for number of workers(cpu) per DataLoader

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        where class_names is a list of target classes.
        Example Usage:
            train_dataloader, test_dataloader, class_names = \
                = create_dataloaders(train_dir=path/to/train_dir,
                                    test_dir=path/to/test_dir,
                                    transform=some_transform,
                                    batch_size=32,
                                    num_workers=4)
    """

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class naems
    class_names = train_data.classes

    # Turn the iimages into dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names