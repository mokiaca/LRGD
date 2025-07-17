"""
Download by running:
```
mkdir -p ~/datasets/cbsd68
wget https://huggingface.co/datasets/deepinv/CBSD68/resolve/main/CBSD68.tar.gz -P ~/datasets/cbsd68
cd ~/datasets/cbsd68 && tar -xvf CBSD68.tar.gz && rm CBSD68.tar.gz && mv CBSD68/0/* . && rm -r CBSD68
```
"""

import os
from typing import Tuple

from omegaconf import DictConfig
from torch.utils.data import Dataset


class CBSD68(Dataset):
    def __init__(self, image_dir: str):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".png")
        ]
        assert len(self.image_paths) == 68

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        return {"image_path": self.image_paths[idx]}


def get_datasets(cfg: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Example:
    ```
    _, val_set, _ = get_datasets(cfg)
    ```
    """
    path = "~/datasets/cbsd68"
    try:
        path = cfg.dataset.path or path
    except AttributeError:
        pass
    path = os.path.expanduser(path)

    ds = CBSD68(path)

    return None, ds, None


if __name__ == "__main__":
    train_set, val_set, _ = get_datasets(DictConfig({}))
    print("Train set:", train_set)
    print("Validation set:", val_set)
