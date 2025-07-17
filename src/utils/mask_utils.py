import random

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import sobel


def generate_random_sampling_mask(
    image: Image.Image, sampling_rate: float
) -> Image.Image:
    """
    Generate a random sampling mask for the given image based on the sampling rate.

    Parameters:
        image (PIL.Image): The input image.
        sampling_rate (float): The sampling rate, between 0 and 1.

    Returns:
        PIL.Image: The mask image with the same size and channels as the input image.
    """
    # Ensure the sampling rate is between 0 and 1
    if not (0 <= sampling_rate <= 1):
        raise ValueError("Sampling rate must be between 0 and 1")

    # Convert image to numpy array
    img_array = np.array(image)

    # Create a mask array with the same shape as the image
    mask_array = np.zeros(img_array.shape, dtype=np.uint8)

    # Generate the mask by randomly setting pixels to 255 (white)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if random.random() > sampling_rate:
                mask_array[i, j] = 255

    # Convert the mask array back to a PIL image
    mask_image = Image.fromarray(mask_array)

    return mask_image


def generate_gradient_based_sampling_mask(
    image: Image.Image, sampling_rate: float
) -> Image.Image:
    # Ensure the sampling rate is between 0 and 1
    if not (0 <= sampling_rate <= 1):
        raise ValueError("Sampling rate must be between 0 and 1")

    # Convert image to numpy array
    image = np.array(image)

    # Check if the image is colorful
    if len(image.shape) == 3:
        # For colorful images
        gradients = []
        for channel in range(image.shape[2]):
            grad_x = sobel(image[:, :, channel], axis=0)
            grad_y = sobel(image[:, :, channel], axis=1)
            gradients.append(np.sqrt(grad_x**2 + grad_y**2))

        # Choose the maximum gradient magnitude across all channels
        gradient_magnitude = np.max(gradients, axis=0)
    else:
        # For grayscale images
        grad_x = sobel(image, axis=0)
        grad_y = sobel(image, axis=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize the gradient magnitude
    gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (
        gradient_magnitude.max() - gradient_magnitude.min()
    )
    gradient_magnitude = gradient_magnitude.astype(np.float64)

    # Calculate the sampling probability based on the gradient magnitude
    sampling_prob = gradient_magnitude / gradient_magnitude.sum()

    # Do sampling based on the probability
    num_samples = int(sampling_rate * image.shape[0] * image.shape[1])
    sampled_indices = np.random.choice(
        image.shape[0] * image.shape[1],
        size=num_samples,
        p=sampling_prob.flatten(),
        replace=False,
    )

    # Create a sampling mask
    sampling_mask = np.zeros(image.shape[:2], dtype=bool)
    sampling_mask.flat[sampled_indices] = True

    mask_image = Image.fromarray(~sampling_mask)
    return mask_image


def generate_saliency_based_sampling_mask(
    image: Image.Image, sampling_rate: float
) -> Image.Image:
    """The Best!"""
    # Ensure the sampling rate is between 0 and 1
    if not (0 <= sampling_rate <= 1):
        raise ValueError("Sampling rate must be between 0 and 1")

    # Convert PIL Image to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Initialize OpenCV's saliency detector
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    # Compute the saliency map
    (success, saliency_map) = saliency.computeSaliency(image)
    if not success:
        raise RuntimeError("Failed to compute saliency map")

    # Normalize the saliency map
    saliency_map = (saliency_map - saliency_map.min()) / (
        saliency_map.max() - saliency_map.min()
    )
    saliency_map = saliency_map.astype(np.float64)

    # Calculate the sampling probability based on the saliency map
    sampling_prob = saliency_map / saliency_map.sum()

    # Do sampling based on the probability
    num_samples = int(sampling_rate * image.shape[0] * image.shape[1])
    sampled_indices = np.random.choice(
        image.shape[0] * image.shape[1],
        size=num_samples,
        p=sampling_prob.flatten(),
        replace=False,
    )

    # Create a sampling mask
    sampling_mask = np.zeros(image.shape[:2], dtype=bool)
    sampling_mask.flat[sampled_indices] = True

    mask_image = Image.fromarray(~sampling_mask)
    return mask_image


def apply_mask_to_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Apply a mask to the given image. The mask should be the same size as the image.

    Parameters:
        image (PIL.Image): The input image.
        mask (PIL.Image): The mask image.

    Returns:
        PIL.Image: The masked image.
    """
    # Ensure the image and mask have the same size
    if image.size != mask.size:
        raise ValueError("Image and mask must have the same size")

    # Convert image to RGBA if not already
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Ensure the mask is in grayscale mode and convert to binary format (black and white only)
    if mask.mode != "L":
        mask = mask.convert("L")

    # Convert image and mask to numpy arrays
    img_array = np.array(image)
    mask_array = np.array(mask)

    # Ensure the mask is in binary format (black and white only)
    mask_array = 1 - mask_array // 255

    # Apply the mask to the image
    img_array[..., :] = img_array[..., :] * mask_array[:, :, None]

    # Convert the result back to a PIL image
    masked_image = Image.fromarray(img_array)

    return masked_image
