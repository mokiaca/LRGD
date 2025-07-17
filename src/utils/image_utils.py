from io import BytesIO

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def inpaint_image(
    masked_image: Image.Image, mask: None | Image.Image = None, fast: bool = True
) -> Image.Image:
    """
    Inpaint the masked areas of an image using the provided mask.
    Parameters:
        masked_image (PIL.Image): The masked image.
        mask (PIL.Image): The mask image.
    Returns:
        PIL.Image: The inpainted image.
    """
    # Convert PIL images to numpy arrays
    masked_img_array = np.array(masked_image)

    if mask is None:
        # Get the mask from the masked image
        mask_array = (masked_img_array[:, :, 3] == 0).astype(np.uint8) * 255
    else:
        mask_array = np.array(mask)

    # binary mask: 0 for black, 1 for white
    mask_array = (mask_array > 127).astype(np.uint8)

    # Ensure the mask is single channel and binary
    if len(mask_array.shape) == 3:
        mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)

    # RGBA -> RGB
    if masked_img_array.shape[2] == 4:
        masked_img_array = cv2.cvtColor(masked_img_array, cv2.COLOR_RGBA2RGB)

    if np.all(mask_array == 0):  # if the mask is all black, no need to process
        return Image.fromarray(masked_img_array)
    if np.all(mask_array == 1):  # if the mask is all white, return the original image
        return Image.fromarray(masked_img_array)

    # Inpaint the image using the mask
    # inpaintRadius set to 2 for faster processing but lower quality
    inpainted_img_array = cv2.inpaint(
        masked_img_array,
        mask_array,
        inpaintRadius=2 if fast else 3,
        flags=cv2.INPAINT_TELEA,
    )

    return Image.fromarray(inpainted_img_array)


def canny(
    image: Image.Image, threshold1: int = 100, threshold2: int = 200
) -> Image.Image:
    """
    Apply Canny edge detection to the input image.

    Args:
        image (PIL.Image.Image): The input image.
        threshold1 (int): The first threshold for the hysteresis procedure.
        threshold2 (int): The second threshold for the hysteresis procedure.

    Returns:
        PIL.Image.Image: The image after applying Canny edge detection.
    """
    image = np.array(image)
    image = cv2.Canny(image, threshold1, threshold2)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


def save_image_tensor(input_tensor: torch.Tensor, filename: str) -> None:
    """
    Save an image tensor to a file.

    Args:
        input_tensor (torch.Tensor): The input image tensor.
        filename (str): The filename to save the image to.

    Raises:
        AssertionError: If the input tensor does not have the expected shape.

    """
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    assert len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device("cpu"))
    # input_tensor = unnormalize(input_tensor)
    input_tensor = input_tensor.squeeze()
    input_tensor = (
        input_tensor.mul_(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .type(torch.uint8)
        .numpy()
    )  # [0,1] -> [0,255], CHW -> HWC, tensor -> numpy
    im = Image.fromarray(input_tensor)
    im.save(fp=filename + ".jpg")


def resize_tensor_images(images: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Resize a batch of images or a single image to the specified target shape.

    Args:
        images (torch.Tensor): The input tensor containing the images.
        target_shape (tuple): The target shape to resize the images to.

    Returns:
        torch.Tensor: The resized images.

    Raises:
        ValueError: If the target_shape is not a tuple or a list of tuples.

    """
    if isinstance(target_shape, tuple):
        resize_transform = transforms.Resize(target_shape)
        if len(images.shape) == 4:
            # for batch of images
            resized_images = torch.stack([resize_transform(img) for img in images])
        else:
            # for single image
            resized_images = resize_transform(images)
    elif isinstance(target_shape, list):
        assert len(images.shape) == 4
        resized_images = torch.stack(
            [transforms.Resize(shape)(img) for shape, img in zip(target_shape, images)]
        )
    else:
        raise ValueError("target_shape must be a tuple or a list of tuples")

    return resized_images


def convert_pil_to_tensor(image: Image.Image | torch.Tensor) -> torch.Tensor:
    """
    Convert a PIL image to a tensor.

    Args:
        image (Image.Image): The input PIL image.

    Returns:
        torch.Tensor: The converted tensor.

    """
    if isinstance(image, torch.Tensor):
        return image

    transform = transforms.ToTensor()
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor


def convert_tensor_to_pil(image_tensor: torch.Tensor | Image.Image) -> Image.Image:
    """
    Convert a tensor to a PIL image.

    Args:
        image_tensor (torch.Tensor): The input tensor.

    Returns:
        Image.Image: The converted PIL image.

    """
    if isinstance(image_tensor, Image.Image):
        return image_tensor

    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()

    image_tensor = image_tensor.squeeze()

    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.mul(255).clamp(0, 255).byte()

    if image_tensor.size(0) == 1:
        # Grayscale: (1, H, W) -> (H, W)
        image_tensor = image_tensor.squeeze(0)
        image = Image.fromarray(image_tensor.numpy(), mode="L")
    else:
        # Color: (3, H, W) -> (H, W, 3)
        image_tensor = image_tensor.permute(1, 2, 0)
        image = Image.fromarray(image_tensor.numpy(), mode="RGB")

    return image


def compress_image_to_quality_jpeg(
    image: Image.Image, target_quality: int
) -> Image.Image:
    """
    Compress an image to a target quality using JPEG compression.

    Args:
        image (Image.Image): The input image.
        target_quality (int): The target quality for JPEG compression.

    Returns:
        Image.Image: The compressed image.

    """
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=target_quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    return compressed_image


def compress_image_to_ratio_jpeg(
    image: Image.Image,
    target_ratio: float,
    max_iterations: int = 10,
    tolerance: float = 0.001,
) -> Image.Image:
    """
    Compress an image to a target compression ratio using binary search.

    Args:
        image: PIL Image object to compress
        target_ratio: Target compression ratio (0-1), e.g., 0.05 for 5% of original size
        max_iterations: Maximum number of binary search iterations
        tolerance: Acceptable difference from target ratio

    Returns:
        Compressed PIL Image object

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If target ratio cannot be achieved within max_iterations
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")
    if not 0 < target_ratio < 1:
        raise ValueError("Target ratio must be between 0 and 1")

    # Get original image size
    original_buffer = BytesIO()
    # image.save(original_buffer, format="JPEG", quality=100)
    image.save(original_buffer, format="PNG", optimize=True)
    original_size = len(original_buffer.getvalue())
    # target_size = original_size * target_ratio
    # logger.debug(f"Original size: {original_size:.0f}, target size: {target_size:.0f}")

    # Initialize binary search parameters
    low_quality = 1
    high_quality = 100
    best_image = None
    best_ratio_diff = float("inf")

    for i in range(max_iterations):
        current_quality = (
            (low_quality + high_quality) // 2
            if i > 0
            else max(1, round(target_ratio * 100))
        )

        # Compress with current quality
        jpeg_buffer = BytesIO()
        image.save(jpeg_buffer, format="JPEG", quality=current_quality)
        jpeg_buffer.seek(0)
        compressed_image = Image.open(jpeg_buffer)

        # calculate its reference format size
        reference_buffer = BytesIO()
        compressed_image.save(reference_buffer, format="PNG", optimize=True)

        current_size = len(reference_buffer.getvalue())
        current_ratio = current_size / original_size

        # logger.debug(f"Range: {low_quality} - ({current_quality}) - {high_quality}, current: {current_size:.0f} ({current_ratio})")

        # Check if we're close enough
        ratio_diff = abs(current_ratio - target_ratio)
        if ratio_diff < tolerance:
            # logger.debug(f"Acceptable ratio difference reached: {ratio_diff:.3f} = {current_ratio:.3f} - {target_ratio:.3f}")
            reference_buffer.seek(0)
            return Image.open(reference_buffer)

        # Update best result
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            reference_buffer.seek(0)
            best_image = Image.open(reference_buffer).copy()

        # Adjust search range
        if current_ratio > target_ratio:
            high_quality = current_quality - 1
        else:
            low_quality = current_quality + 1

        # Check if search range is exhausted
        if low_quality > high_quality:
            break

    # If we couldn't achieve the exact target ratio, return the closest result
    if best_image is None:
        raise RuntimeError(
            f"Could not achieve target ratio {target_ratio} within {max_iterations} iterations"
        )

    # logger.debug(
    #     f"Achieved compression ratio: {best_ratio_diff + target_ratio:.3f} (target: {target_ratio})"
    # )
    return best_image
