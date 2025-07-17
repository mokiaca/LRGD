import os
import sys
import time
from typing import List

import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig

from utils.image_utils import canny, compress_image_to_quality_jpeg
from utils.mask_utils import apply_mask_to_image, generate_saliency_based_sampling_mask
from utils.walsh_cs_utils import walsh_hadamard_encode

from .third_party_models.clip_ci import load_ci_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class Transmitter(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Transmitter, self).__init__()
        self.cs_method: str = cfg.cs_method
        self.sampling_rate: float = cfg.sampling.rate
        self.if_enable_clip: bool = cfg.clip_model.enable
        self.if_enable_clip_neg: bool = cfg.clip_model.enable_negative
        self.if_enable_contour: bool = cfg.contour.enable
        self.canny_threshold1: float = cfg.contour.canny_threshold1
        self.canny_threshold2: float = cfg.contour.canny_threshold2
        self.ci_fast_mode: bool = cfg.clip_model.fast_mode
        logger.info(f"CS method: {self.cs_method}")
        logger.info(f"Sampling rate: {self.sampling_rate}")

        # Load the model
        if self.if_enable_clip:
            self.ci = load_ci_model(cfg.clip_model.model_name)

    def forward(self, image) -> tuple[dict, dict]:
        if isinstance(image, list):
            return [self.forward(img) for img in image]

        times: dict = {}

        start_time = time.time()
        desc, neg_desc = self.generate_clip_description(image)
        times["text"] = time.time() - start_time

        start_time = time.time()
        contour = self.generate_contour(image)
        times["edge"] = time.time() - start_time

        ret = {
            "trans_method": self.cs_method,
            "clip_description": desc,
            "clip_neg_description": neg_desc,
            "contour": contour,
        }

        if self.cs_method == "pixel":
            start_time = time.time()
            ret["sampled_pixels"] = self.sample_pixels(image)
            times["sparse"] = time.time() - start_time
        elif self.cs_method == "jpeg":
            ret["image"] = compress_image_to_quality_jpeg(
                image, round(self.sampling_rate * 100)
            )
            ret["compression_ratio"] = self.sampling_rate
        elif self.cs_method == "walsh":
            compressed_data, metadata = walsh_hadamard_encode(image, self.sampling_rate)
            metadata["compressed_data"] = compressed_data
            ret["walsh_metadata"] = metadata
        else:
            raise ValueError(f"Unknown CS method: {self.cs_method}")

        return ret, times

    def sample_pixels(self, image):
        random_mask = generate_saliency_based_sampling_mask(image, self.sampling_rate)
        masked_image = apply_mask_to_image(image, random_mask)
        return masked_image

    def generate_clip_description(self, image):
        if not self.if_enable_clip:
            return None, None

        if self.ci_fast_mode:
            desc = self.ci.interrogate_fast(image)
        else:
            desc = self.ci.interrogate_classic(image)
        neg_desc = (
            self.ci.interrogate_negative(image) if self.if_enable_clip_neg else None
        )
        return desc, neg_desc

    def generate_contour(self, image):
        if not self.if_enable_contour:
            return None

        contour = canny(image, self.canny_threshold1, self.canny_threshold2)
        contour = contour.convert("1")  # Convert to binary image
        return contour
