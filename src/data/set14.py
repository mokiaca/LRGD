"""
Auto download
"""

from typing import Tuple

import datasets
from omegaconf import DictConfig
from torch.utils.data import Dataset


def get_datasets(cfg: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Example:
    ```
    _, val_set, _ = get_datasets(cfg)
    ```
    """

    ds = datasets.load_dataset("eugenesiow/Set14", "bicubic_x2")

    return None, ds["validation"], None


if __name__ == "__main__":
    train_set, val_set, _ = get_datasets(DictConfig({}))
    print("Train set:", train_set)
    print("Validation set:", val_set)
