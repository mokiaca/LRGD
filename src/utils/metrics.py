import math

import piqa
import torch
from PIL import Image

from .image_utils import convert_pil_to_tensor, resize_tensor_images

psnr_model = None
ssim_model = None
ms_ssim_model = None
lpips_model = None
fid_model = None


def calculate_all_metrics(
    img1: torch.Tensor | list | Image.Image, img2: torch.Tensor
) -> dict:
    if isinstance(img1, list) and isinstance(img2, list):
        # Calculate average metrics for a list of images
        assert len(img1) == len(img2), "Both lists must have the same length"
        metrics = [calculate_all_metrics(i1, i2) for i1, i2 in zip(img1, img2)]
        avg_metrics = {k: sum(m[k] for m in metrics) / len(metrics) for k in metrics[0]}
        return avg_metrics

    if isinstance(img1, Image.Image):
        img1 = convert_pil_to_tensor(img1)
    if isinstance(img2, Image.Image):
        img2 = convert_pil_to_tensor(img2)
    return {
        "psnr": calculate_psnr(img1, img2),
        "ssim": calculate_ssim(img1, img2),
        "ms_ssim": calculate_ms_ssim(img1, img2),
        "lpips": calculate_lpips(img1, img2),
    }


def calculate_psnr(
    img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1
) -> float:
    """
    calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1 (torch.Tensor): The first image tensor.
        img2 (torch.Tensor): The second image tensor.
        data_range (float): The data range of the input images (default: 255).

    Returns:
        float: The PSNR value (dB).
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100
    psnr_value = 20 * math.log10(data_range / math.sqrt(mse))
    return psnr_value


def calculate_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
) -> float:
    assert img1.device == img2.device, "Both images must be on the same device"

    global ssim_model
    if ssim_model is None:
        ssim_model = piqa.SSIM().to(img1.device)

    score = ssim_model(img1, img2).item()
    return score


def calculate_ms_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
) -> float:
    assert img1.device == img2.device, "Both images must be on the same device"

    global ms_ssim_model
    if ms_ssim_model is None:
        ms_ssim_model = piqa.MS_SSIM().to(img1.device)

    if img1.shape[-1] < 256 or img1.shape[-2] < 256:
        img1 = resize_tensor_images(img1, (256, 256))
        img2 = resize_tensor_images(img2, (256, 256))
    score = ms_ssim_model(img1.clamp(0, 1), img2.clamp(0, 1)).item()
    return score


def calculate_lpips(img1: torch.Tensor, img2: torch.Tensor) -> float:
    assert img1.device == img2.device, "Both images must be on the same device"

    global lpips_model
    if lpips_model is None:
        lpips_model = piqa.LPIPS().to(img1.device)

    score = lpips_model(img1, img2).item()
    return score


def calculate_fid(imgs1: list, imgs2: list) -> float:
    """
    Can only be calculated at the end for all images in one go.
    """
    for i, (img1, img2) in enumerate(zip(imgs1, imgs2)):
        if isinstance(img1, Image.Image):
            imgs1[i] = convert_pil_to_tensor(img1)
        if isinstance(img2, Image.Image):
            imgs2[i] = convert_pil_to_tensor(img2)

    global fid_model
    if fid_model is None:
        fid_model = piqa.FID().to(imgs1[0].device)

    imgs1 = torch.cat([fid_model.features(img) for img in imgs1], dim=0)
    imgs2 = torch.cat([fid_model.features(img) for img in imgs2], dim=0)

    score = fid_model(imgs1, imgs2).item()
    return score


if __name__ == "__main__":
    import requests
    from PIL import ImageFilter
    from torchvision.transforms.functional import to_tensor

    # Load original image from URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    original_image = Image.open(requests.get(url, stream=True).raw)

    # Convert image to tensor for processing
    original_tensor = to_tensor(original_image).unsqueeze(0)  # Add batch dimension

    # Create a blurred version of the image (e.g., Gaussian blur)
    blurred_image = original_image.filter(ImageFilter.GaussianBlur(radius=5))
    blurred_tensor = to_tensor(blurred_image).unsqueeze(0)  # Add batch dimension

    # calculate PSNR between original and blurred images
    psnr_value = calculate_psnr(original_tensor, blurred_tensor)
    print(f"PSNR: {psnr_value:.5g} dB")

    # calculate SSIM between original and blurred images
    ssim_value = calculate_ssim(original_tensor, blurred_tensor)
    print(f"SSIM: {ssim_value:.5g}")
    ssim_new = piqa.SSIM()(original_tensor, blurred_tensor)

    # calculate MS-SSIM between original and blurred images
    ms_ssim_value = calculate_ms_ssim(original_tensor, blurred_tensor)
    print(f"MS-SSIM: {ms_ssim_value:.5g}")
    ms_ssim_new = piqa.MS_SSIM()(original_tensor, blurred_tensor)
    print(f"MS_SSIM_new: {ms_ssim_new:.5g}")

    # calculate LPIPS between original and blurred images
    lpips_value = calculate_lpips(original_tensor, blurred_tensor)
    print(f"LPIPS: {lpips_value:.5g}")

    # Show the original and blurred images
    original_image.show(title="Original Image")
    blurred_image.show(title="Blurred Image")
