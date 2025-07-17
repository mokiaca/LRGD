from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet import retrieve_timesteps
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import retrieve_latents
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version


def svd_low_rank_decomposition(
    tensor, rank=8, info_percent=None, rank_percent=None, if_return_u_r=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform low-rank decomposition on a given tensor.

    Args:
        tensor (torch.Tensor): The input tensor to be decomposed. It should be either 3D or 4D.
        rank (int, optional): The desired rank of the decomposition. Defaults to 8.
        info_percent (float, optional): The percentage of information to be retained in the low-rank approximation.
            Defaults to 0.95. If `rank` is not provided, this parameter will be used to determine the rank.
        if_return_u_r (bool, optional): Whether to return the left singular vectors and the right singular vectors.

    Returns:
        torch.Tensor: The low-rank approximation of the input tensor.
        torch.Tensor: The fine features obtained by subtracting the low-rank approximation from the input tensor.
        (optional) torch.Tensor: The left singular vectors of the low-rank approximation.
    """
    # Ensure the tensor is of type float32 or float64
    if tensor.dtype == torch.float16:
        tensor = tensor.float()

    if rank == 0:
        return tensor, torch.zeros_like(tensor)

    # Check if the input is 3D or 4D
    if tensor.dim() == 3:
        dim = 3
        tensor = tensor.unsqueeze(0)  # Add a batch dimension
    elif tensor.dim() == 4:
        dim = 4
    else:
        raise ValueError("Tensor must be either 3D or 4D")

    # Reshape the tensor for SVD
    B, C, H, W = tensor.shape
    tensor_reshaped = tensor.view(B * C, H, W)

    # Perform SVD on the reshaped tensor
    u, s, vh = torch.linalg.svd(tensor_reshaped, full_matrices=False)

    if rank is None and info_percent is not None:
        # use info_percent to determine the rank
        total_energy = torch.sum(s**2, dim=1)
        cumulative_energy = torch.cumsum(s**2, dim=1)
        relative_energy = cumulative_energy / total_energy.unsqueeze(1)
        # Find the number of components needed to retain the specified percentage of information
        num_components = torch.sum(relative_energy < info_percent, dim=1) + 1
        # Get the mean number of components across all B*C slices
        rank = num_components.float().mean().int().item()
    elif rank_percent is not None:
        rank = int(rank_percent * min(H, W))

    # Keep only the top-rank components
    u = u[:, :, :rank]  # [B*C, H, rank]
    s = s[:, :rank]  # [B*C, rank]
    vh = vh[:, :rank, :]  # [B*C, rank, W]

    # Reconstruct the low-rank approximation
    low_rank_tensor = torch.bmm(u, torch.bmm(torch.diag_embed(s), vh))  # [B*C, H, W]

    # Reshape the low-rank tensor back to the original shape
    low_rank_tensor = low_rank_tensor.view(B, C, H, W)
    if if_return_u_r:
        u = u.view(B, C, H, -1)

    # Calculate fine features
    fine_tensor = tensor - low_rank_tensor  # [B, C, H, W]

    # If the input was 3D, remove the added batch dimension
    if dim == 3:
        low_rank_tensor = low_rank_tensor.squeeze(0)
        fine_tensor = fine_tensor.squeeze(0)
        if if_return_u_r:
            u = u.squeeze(0)

    if if_return_u_r:
        return low_rank_tensor, fine_tensor, u

    return low_rank_tensor, fine_tensor


def orthogonal_projection(
    z_t: torch.Tensor, u_r: torch.Tensor, check_properties: bool = False
):
    B, D, H, W = z_t.shape
    z_t = z_t.view(B * D, H, W)
    u_r = u_r.view(B * D, H, -1)

    if check_properties:
        # Check orthogonality
        orthogonality = torch.abs(
            u_r.transpose(-1, -2) @ u_r - torch.eye(u_r.size(-1), device=u_r.device)
        )
        assert torch.allclose(
            orthogonality, torch.zeros_like(orthogonality), atol=1e-3
        ), "The columns of u_r are not orthogonal"

        # Check if the columns are unit vectors
        norms = torch.norm(u_r, dim=1)
        assert torch.allclose(
            norms, torch.ones_like(norms), atol=1e-3
        ), "The columns of u_r are not unit vectors"

    projection = u_r @ u_r.transpose(-1, -2)

    if check_properties:
        # Check the properties of the projection matrix
        assert torch.allclose(
            projection, projection.transpose(-1, -2), atol=1e-3
        ), "Projection matrix is not symmetric"
        assert torch.allclose(
            projection @ projection, projection, atol=1e-3
        ), "Projection matrix is not idempotent"

    projected = projection @ z_t
    projected = projected.view(B, D, H, W)
    return projected


class Schedulers:
    @staticmethod
    def two_stage_schedule(t, T, start_value=0, mid_value=1, end_value=0) -> float:
        """
        0 -> 1 -> 0 /\
        """
        if t < T // 2:
            return start_value + (mid_value - start_value) * t / (T // 2)
        else:
            return mid_value + (end_value - mid_value) * (t - T // 2) / (T // 2)

    @staticmethod
    def sqrt_two_stage_schedule(t, T, start_value=0, mid_value=1, end_value=0) -> Tuple:
        """
        0 -> 1 -> 0
        """
        if t < T // 2:
            value = start_value + (mid_value - start_value) * t / (T // 2)
        else:
            value = mid_value + (end_value - mid_value) * (t - T // 2) / (T // 2)
        return value**0.5, (1 - value) ** 0.5

    @staticmethod
    def half_cosine_schedule(t, T, start_value=0, end_value=1) -> float:
        # 0 -> 1
        cos_val = np.cos(np.pi * t / T) / 2 + 0.5
        return start_value + (end_value - start_value) * cos_val

    @staticmethod
    def full_cosine_schedule(t, T, start_value=0, mid_value=1) -> float:
        """0 -> 1 -> 0"""
        cos_val = -np.cos(np.pi * 2 * t / T) / 2 + 0.5
        return start_value + (mid_value - start_value) * cos_val

    @staticmethod
    def half_sine_schedule(t, T, start_value=0, mid_value=1):
        """
        The best schedule so far.
        0 -> 1 -> 0
        """
        sin_val = np.sin(t / T * np.pi)
        return start_value + (mid_value - start_value) * sin_val

    @staticmethod
    def full_sine_schedule(t, T, min_value=0, max_value=1):
        # 0.5 -> 1 -> 0.5 -> 0 -> 0.5
        sin_val = np.sin(t / T * np.pi * 2) / 2 + 0.5
        return min_value + (max_value - min_value) * sin_val

    @staticmethod
    def annealing_schedule(t, T, start_value=1, end_value=0, temperature=1.0):
        """
        Exponential annealing schedule
        1 -> 1/e
        """
        x = t / T
        return end_value + (start_value - end_value) * np.exp(-x / temperature)

    @staticmethod
    def one_cycle_schedule(t, T, initial_v=0.0, max_v=1.0, min_v=0.0, warmup_ratio=0.5):
        """
        One cycle schedule. Not very good.
        0.1 -> 1.0 -> 0.0
        """
        warmup = int(warmup_ratio * T)
        if t <= warmup:
            # Warmup
            return initial_v + (max_v - initial_v) * t / warmup
        else:
            # Annealing
            # return max_v + (min_v - max_v) * (t - warmup) / (T - warmup)
            return (
                min_v
                + (max_v - min_v)
                * (1 + np.cos(np.pi * (t - warmup) / (T - warmup)))
                / 2
            )


class LowRankGuidedSDCNP(StableDiffusionControlNetPipeline):

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,  # ControlNet input
        guide_image: PipelineImageInput = None,  # Our method: low-rank guided diffusion use image
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        rank: int = 24,  # Our method: low-rank guided diffusion rank
        info_percent: float = None,  # Our method: low-rank guided diffusion info percent
        rank_percent: float = None,  # Our method: low-rank guided diffusion rank percent
        ortho_projection: bool = False,  # Our method: low-rank guided diffusion orthogonal projection
        smooth_transition: bool = False,  # Our method: low-rank guided diffusion smooth transition
        sampling_rate: float = 0.0,  # Our method: use rate to determine the smooth param
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        **kwargs,
    ):

        controlnet = (
            self.controlnet._orig_mod
            if is_compiled_module(self.controlnet)
            else self.controlnet
        )

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(
            control_guidance_end, list
        ):
            control_guidance_start = len(control_guidance_end) * [
                control_guidance_start
            ]
        elif not isinstance(control_guidance_end, list) and isinstance(
            control_guidance_start, list
        ):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(
            control_guidance_end, list
        ):
            mult = (
                len(controlnet.nets)
                if isinstance(controlnet, MultiControlNetModel)
                else 1
            )
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(
            controlnet_conditioning_scale, float
        ):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
                controlnet.nets
            )

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # MARK: Rough feature of the guide image
        guide_tensor = self.image_processor.preprocess(guide_image).to(device=device)
        guide_latents = self._encode_vae_image(guide_tensor, generator)
        low_rank_guided_latents, _, u_r = svd_low_rank_decomposition(
            guide_latents,
            rank=rank,
            if_return_u_r=True,
            info_percent=info_percent,
            rank_percent=rank_percent,
        )
        height, width = guide_tensor.shape[-2:]
        # END MARK

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = (
                self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                if image is not None
                else None
            )
            # height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            # Nested lists as ControlNet condition
            if isinstance(image[0], list):
                # Transpose the nested image list
                image = [list(t) for t in zip(*image)]

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            # height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 7.2 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(
                keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
            )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (
                    is_unet_compiled and is_controlnet_compiled
                ) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(
                        control_model_input, t
                    )
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [
                        c * s
                        for c, s in zip(
                            controlnet_conditioning_scale, controlnet_keep[i]
                        )
                    ]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = (
                    self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=image,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )
                    if image is not None
                    else (None, None)
                )

                if guess_mode and self.do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = (
                        [
                            torch.cat([torch.zeros_like(d), d])
                            for d in down_block_res_samples
                        ]
                        if down_block_res_samples is not None
                        else None
                    )
                    mid_block_res_sample = (
                        torch.cat(
                            [
                                torch.zeros_like(mid_block_res_sample),
                                mid_block_res_sample,
                            ]
                        )
                        if mid_block_res_sample is not None
                        else None
                    )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # MARK: Use low-rank guided features to guide the diffusion
                if not ortho_projection:
                    low_rank_latents, high_freq_latents = svd_low_rank_decomposition(
                        latents, rank=rank, info_percent=info_percent
                    )  # w/o orthogonal projection
                if smooth_transition:
                    alpha = Schedulers.half_sine_schedule(
                        i, num_inference_steps, sampling_rate, 1.0
                    )
                    alpha_prime = 1 - alpha
                else:
                    alpha = 1.0  # for comparison / ablation study
                    alpha_prime = 1 - alpha
                if not ortho_projection:
                    latents = (
                        alpha * low_rank_guided_latents
                        + alpha_prime * low_rank_latents
                        + high_freq_latents
                    )  # w/o orthogonal projection
                else:
                    latents = (
                        latents
                        - (1 - alpha_prime) * orthogonal_projection(latents, u_r)
                        + alpha * low_rank_guided_latents
                    )  # w/ orthogonal projection
                # END MARK

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        dtype = image.dtype
        if self.vae.config.force_upcast:
            image = image.float()
            self.vae.to(dtype=torch.float32)

        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(
                    self.vae.encode(image[i : i + 1]), generator=generator[i]
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(
                self.vae.encode(image), generator=generator
            )

        if self.vae.config.force_upcast:
            self.vae.to(dtype)

        image_latents = image_latents.to(dtype)
        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents
