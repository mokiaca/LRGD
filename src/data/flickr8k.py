"""
Download by running:
```
mkdir -p ~/datasets/flickr8k
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip -P ~/datasets/flickr8k
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip -P ~/datasets/flickr8k
```
"""

import os
from typing import Tuple

import datasets
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset


def get_datasets(cfg: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Example:
    ```
    train_set, val_set, test_set = get_datasets(cfg)
    ```
    """
    path = "~/datasets/flickr8k"
    try:
        path = cfg.dataset.path or path
    except AttributeError:
        pass
    path = os.path.expanduser(path)

    ds = datasets.load_dataset("atasoglu/flickr8k-dataset", data_dir=path)

    return ds["train"], ds["validation"], ds["test"]


def get_all_in_one_dataset(cfg: DictConfig) -> Dataset:
    train, val, test = get_datasets(cfg)

    all_in_one = ConcatDataset([train, val, test])

    return all_in_one


def get_all_in_one_dataloader(
    cfg: DictConfig, num_workers: int = 0, pin_memory: bool = True
):
    all_in_one = get_all_in_one_dataset(cfg)

    all_in_one_loader = DataLoader(
        all_in_one,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return all_in_one_loader


def get_dataloaders(cfg: DictConfig, num_workers: int = 0, pin_memory: bool = True):
    train, val, test = get_datasets(cfg)

    train_loader = DataLoader(
        train,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_set, val_set, _ = get_datasets(DictConfig({}))
    print("Train set:", train_set)
    print("Validation set:", val_set)
