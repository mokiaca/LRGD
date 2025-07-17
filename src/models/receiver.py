import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from PIL import Image

from utils.image_utils import inpaint_image
from utils.walsh_cs_utils import walsh_hadamard_decode

from .third_party_models.guide_sdcnp import LowRankGuidedSDCNP
from .third_party_models.load_sd_pipe import load_sd_pipe
from .traditional.interpolate import interpolate_sparse_image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class Receiver(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Receiver, self).__init__()
        self.method: str = cfg.method
        if self.method == "sd":
            self.num_inference_step: int = cfg.stable_diffusion.num_inference_step or 20
            self.rank: int = cfg.stable_diffusion.rank
            self.info_percent: float = cfg.stable_diffusion.info_percent
            self.rank_percent: float = cfg.stable_diffusion.rank_percent
            self.ortho_projection: bool = cfg.stable_diffusion.ortho_projection
            self.smooth_transition: bool = cfg.stable_diffusion.smooth_transition
            self.generator = torch.Generator(device="cpu").manual_seed(1)
            logger.info(f"Number of inference steps: {self.num_inference_step}")
            logger.info(f"Rank: {self.rank}")
        else:
            logger.info(f"*** Receiver method: {self.method} ***")

        # Load the model
        if cfg.stable_diffusion.enable:
            load_pipe = {
                "sd1.5": load_sd_pipe,
            }[cfg.stable_diffusion.model_name]
            self.pipe: LowRankGuidedSDCNP = load_pipe(
                method=cfg.stable_diffusion.method,
                use_lcm_lora=cfg.stable_diffusion.use_lcm_lora,
            )
            logger.info(f"Loaded model: {cfg.stable_diffusion.model_name}")
        else:
            self.pipe = None

    def freeze_pipe(self):
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.controlnet.requires_grad_(False)

    def forward(self, transmitter_output, get_inpaint=False):
        if isinstance(transmitter_output, list):
            return [self.forward(i, get_inpaint) for i in transmitter_output]

        times: dict = {}

        trans_method: str = transmitter_output["trans_method"]
        clip_desc: str = transmitter_output["clip_description"]
        clip_neg_desc: str = transmitter_output["clip_neg_description"]
        contour: Image.Image = transmitter_output["contour"]

        if trans_method == "pixel":
            sampled_pixels: Image.Image = transmitter_output["sampled_pixels"]
            sampling_ratio: float = 1 - transparent_pixel_ratio(sampled_pixels)
            w, h = sampled_pixels.width, sampled_pixels.height
        elif trans_method == "jpeg":
            low_quality_image: Image.Image = transmitter_output["image"]
            sampling_ratio = transmitter_output["compression_ratio"]
            w, h = low_quality_image.width, low_quality_image.height
        elif trans_method == "walsh":
            walsh_metadata = transmitter_output["walsh_metadata"]
            low_quality_image = walsh_hadamard_decode(
                walsh_metadata["compressed_data"],
                walsh_metadata,
            )
            sampling_ratio = walsh_metadata["sampling_ratio"]
            h, w = walsh_metadata["original_size"]

        if self.method == "bi":  # bilinear interpolation
            assert (
                trans_method == "pixel"
            ), "Only pixel cs mode is supported for Bilinear Interpolation"
            image = low_quality_image = interpolate_sparse_image(sampled_pixels)

        elif self.method == "jpeg":  # jpeg compression
            assert trans_method == "jpeg", "Only jpeg cs mode is supported for JPEG"
            image = low_quality_image

        elif self.method == "walsh":
            assert (
                trans_method == "walsh"
            ), "Only walsh cs mode is supported for Walsh Hadamard"
            image = low_quality_image

        elif self.method == "sd":
            if trans_method == "pixel":
                start_time = time.time()
                low_quality_image = inpaint_image(sampled_pixels)
                times["pre-inpaint"] = time.time() - start_time

            if self.pipe is not None:
                start_time = time.time()
                with torch.autocast("cuda"):
                    if isinstance(self.pipe, LowRankGuidedSDCNP):
                        image = self.pipe(
                            clip_desc or "",
                            negative_prompt=clip_neg_desc or "",
                            rank=self.rank,
                            info_percent=self.info_percent,
                            rank_percent=self.rank_percent,
                            ortho_projection=self.ortho_projection,
                            smooth_transition=self.smooth_transition,
                            num_inference_steps=self.num_inference_step,
                            generator=self.generator,
                            # eta=1.0, # why I set eta=1.0? It's wrong!
                            image=contour,
                            guide_image=low_quality_image,
                            sampling_rate=sampling_ratio,
                        ).images[0]
                    else:
                        assert (
                            trans_method == "pixel"
                        ), "Only pixel cs mode is supported for this model"
                        # reverse the mask
                        mask = Image.eval(
                            sampled_pixels.split()[-1], lambda a: 255 if a == 0 else 0
                        ).convert("L")
                        image = self.pipe(
                            clip_desc or "",
                            negative_prompt=clip_neg_desc or "",
                            num_inference_steps=self.num_inference_step,
                            generator=self.generator,
                            # eta=1.0,
                            image=low_quality_image,
                            mask_image=mask,
                            control_image=contour,
                        ).images[0]

                    image = image.resize((w, h))

                times["diffusion"] = time.time() - start_time
            else:
                image = low_quality_image
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if get_inpaint:
            return image, low_quality_image, times
        return image, times


def transparent_pixel_ratio(image: Image.Image) -> float:
    if image.mode != "RGBA":
        raise ValueError("Input image must be in RGBA format")

    image_array = np.array(image)
    alpha_channel = image_array[:, :, 3]
    transparent_pixels = np.sum(alpha_channel == 0)
    total_pixels = alpha_channel.size
    ratio = transparent_pixels / total_pixels

    return ratio
