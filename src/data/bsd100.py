"""
Auto download
"""

from typing import Tuple

import datasets
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset


def get_datasets(cfg: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Example:
    ```
    train_set, val_set, _ = get_datasets(cfg)
    ```
    """

    ds = datasets.load_dataset("eugenesiow/BSD100", "bicubic_x2")

    return None, ds["validation"], None


def get_all_in_one_dataset(cfg: DictConfig) -> Dataset:
    train, val, _ = get_datasets(cfg)

    all_in_one = ConcatDataset([train, val])

    return all_in_one


if __name__ == "__main__":
    train_set, val_set, _ = get_datasets(DictConfig({}))
    print("Train set:", train_set)
    print("Validation set:", val_set)
