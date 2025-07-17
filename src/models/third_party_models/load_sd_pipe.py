import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    LCMScheduler,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)

from .guide_sdcnp import LowRankGuidedSDCNP


def load_sd_pipe(method: str, torch_dtype=torch.float16, use_lcm_lora=False):
    """
    Get the Stable Diffusion pipeline with ControlNet for inpainting.
    Ref: https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sd
    """
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float32
    )
    pipe_cls = {
        "inpaint": StableDiffusionControlNetInpaintPipeline,
        "lowrank": LowRankGuidedSDCNP,
    }[method]
    pipe = pipe_cls.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )
    if not use_lcm_lora:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.enable_model_cpu_offload()
    return pipe
