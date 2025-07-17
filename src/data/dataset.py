from omegaconf import DictConfig
from torch.utils.data import Dataset


def get_test_dataset(cfg: DictConfig) -> Dataset:

    name = cfg.dataset.name

    if name == "bsd100":
        from .bsd100 import get_datasets

        _, val, _ = get_datasets(cfg)
        return val

    elif name == "cbsd68":
        from .cbsd68 import get_datasets

        _, val, _ = get_datasets(cfg)
        return val

    elif name == "div2k":
        from .div2k import get_datasets

        _, val, _ = get_datasets(cfg)
        return val

    elif name == "flickr8k":
        from .flickr8k import get_datasets

        _, _, test = get_datasets(cfg)
        return test

    elif name == "set14":
        from .set14 import get_datasets

        _, val, _ = get_datasets(cfg)
        return val

    elif name == "urban100":
        from .urban100 import get_datasets

        _, val, _ = get_datasets(cfg)
        return val
