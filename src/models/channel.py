import io
import sys
import zlib
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from PIL import Image

from utils.qam import qam16ModulationImage, qam16ModulationString, qam16ModulationTensor

# These keys are exempt from adding noise
noise_exempt_keys = [
    "trans_method",
    "compression_ratio",
    "original_size",
    "padded_shape",
    "block_size",
    "sampling_ratio",
    "n_channels",
    "blocks_h",
    "blocks_w",
]


class Channel(nn.Module):
    def __init__(self, cfg: DictConfig = None):
        super(Channel, self).__init__()
        if cfg is None:
            cfg = DictConfig(dict(channel_type="none", snr=None))
        self.chan_type = cfg.channel_type  # "awgn", "none"
        self.snr = cfg.snr
        assert self.chan_type in ["none", "awgn"], "Only AWGN channel is supported."
        logger.info("Built {} channel, SNR {} dB.".format(cfg.channel_type, cfg.snr))

    def gaussian_noise_layer(self, input: str | Image.Image | torch.Tensor):
        if isinstance(input, str):
            return qam16ModulationString(input, snr_db=self.snr)

        elif isinstance(input, Image.Image):
            return qam16ModulationImage(input, snr_db=self.snr)

        elif isinstance(input, torch.Tensor):
            return qam16ModulationTensor(input, snr_db=self.snr)

        elif isinstance(input, np.ndarray):
            return qam16ModulationTensor(
                torch.from_numpy(input), snr_db=self.snr
            ).numpy()

        else:
            # raise ValueError("Unsupported input type.")
            return input

    def calculate_size_KB(self, data, sparse=False):
        """
        Calculate the size of the input data in KB.
        """

        def get_byte_size(obj, sparse=False):
            if obj is None:
                return 0
            if isinstance(obj, int):
                return sys.getsizeof(obj)
            if isinstance(obj, float):
                return sys.getsizeof(obj)
            if isinstance(obj, str):
                return len(obj.encode("utf-8"))
                # return len(obj.encode("ascii"))  # Error
            if isinstance(obj, bytes):
                return sys.getsizeof(obj)
            if isinstance(obj, tuple):
                return sum(get_byte_size(v) for v in obj)
            if isinstance(obj, list):
                return sum(get_byte_size(v) for v in obj)
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            if isinstance(obj, torch.Tensor):
                return obj.element_size() * obj.nelement()
            if isinstance(obj, Image.Image):
                buffer = io.BytesIO()
                if sparse:  # PNG's size is larger than theoretical
                    obj.save(buffer, format="WebP", lossless=True)
                else:
                    obj.save(buffer, format="PNG", optimize=True)
                return len(buffer.getvalue())
            if isinstance(obj, dict):
                return sum(get_byte_size(v, sparse) for _, v in obj.items())
            raise ValueError(f"Unsupported input type: {type(obj)}")

        size = get_byte_size(data, sparse)
        return size / 1024  # Convert to KB

    def forward(
        self, input: list | tuple | dict
    ) -> Tuple[dict, dict] | List[Tuple[dict, dict]]:
        if isinstance(input, list):
            return [self.forward(i) for i in input]
        if isinstance(input, tuple):
            return tuple(self.forward(i) for i in input)

        # Calculate the KB size of the input
        sparse = {k: False for k in input.keys()}
        sparse["sampled_pixels"] = True
        input_sizes = dict(total=self.calculate_size_KB(input, sparse=sparse))
        if "clip_description" in input.keys():
            input_sizes["text"] = self.calculate_size_KB(
                input["clip_description"], sparse=sparse
            )
        if "contour" in input.keys():
            input_sizes["edge"] = self.calculate_size_KB(
                input["contour"], sparse=sparse
            )
        if "sampled_pixels" in input.keys():
            input_sizes["sparse"] = self.calculate_size_KB(
                input["sampled_pixels"], sparse=sparse
            )

        # Add noise to the input objects
        return self.qam(input), input_sizes

    def qam(self, obj: None | dict | list | tuple | str | Image.Image | torch.Tensor):
        """
        Add noise to the input object.
        """
        if obj is None:
            return None
        if self.snr is None or self.snr >= 100:
            return obj
        if isinstance(obj, dict):
            return {
                k: self.qam(v) if k not in noise_exempt_keys else v
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self.qam(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self.qam(i) for i in obj)

        if self.chan_type == "none":
            return obj
        elif self.chan_type == "awgn":
            return self.gaussian_noise_layer(obj)
        else:
            raise ValueError(f"Unsupported channel type {self.chan_type}")


if __name__ == "__main__":
    # Test the channel
    cfg = DictConfig(
        {
            "channel_type": "awgn",  # "awgn", "rayleigh"
            "multiple_snr": 20,
        }
    )
    channel = Channel(cfg)
    text = "Hello, world!"
    for _ in range(5):
        noisy_text = channel.qam(text, 0)
        logger.info(f"Noisy text: {noisy_text}")

    img = Image.open("test.jpg")
    noisy_img = channel.qam(img, 0)
    noisy_img.show()
    noisy_img.save("test_noisy.jpg")
