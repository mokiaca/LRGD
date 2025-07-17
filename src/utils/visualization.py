import math
from typing import List

import matplotlib.pyplot as plt
import torch
from PIL import Image

from .image_utils import convert_tensor_to_pil


def display_images(images: List[Image.Image]):
    """
    Show a list of images in 1 row.
    """
    num_images = len(images)
    _, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))

    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.show()


def arrange_images(images: List[Image.Image], resize_to: int = None) -> Image.Image:
    """
    Arrange multiple images into a single image, placed as close to a square as possible.
    Replace None values with black images.

    Args:
        images (List[Image.Image]): List of PIL images.

    Returns:
        Image.Image: Combined image.
    """
    not_none_images = [img for img in images if img is not None]
    if len(not_none_images) == 0:
        return Image.new("RGB", (resize_to, resize_to), (0, 0, 0))
    image_size = not_none_images[0].size
    images = [
        img if img is not None else Image.new("RGB", image_size, (0, 0, 0))
        for img in images
    ]

    num_images = len(images)
    grid_size = math.ceil(math.sqrt(num_images))
    image_width, image_height = images[0].size
    combined_width = grid_size * image_width
    combined_height = grid_size * image_height

    combined_image = Image.new("RGBA", (combined_width, combined_height))

    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = convert_tensor_to_pil(img)
        row = i // grid_size
        col = i % grid_size
        combined_image.paste(img, (col * image_width, row * image_height))

    if resize_to is not None:
        combined_image = combined_image.resize((resize_to, resize_to))

    return combined_image


def arrange_images_side_by_side(images, separator_width=10):
    """
    Arrange images side by side with a separator in between.

    Args:
        images (List[Image.Image]): List of PIL images.
        separator_width (int): Width of the separator between images.

    Returns:
        Image.Image: Combined image.
    """
    # Filter out None values and replace them with black images of the same size
    images = [
        img if img is not None else Image.new("RGBA", images[0].size, (0, 0, 0, 0))
        for img in images
    ]

    # Determine the height of the output image based on the first image
    image_height = images[0].height
    total_width = sum(img.width for img in images) + separator_width * (len(images) - 1)

    # Create a new blank image with transparent background
    combined_image = Image.new("RGBA", (total_width, image_height), (255, 255, 255, 0))

    # Paste images into the combined image
    current_x = 0
    for img in images:
        combined_image.paste(img, (current_x, 0))
        current_x += img.width + separator_width

    return combined_image


if __name__ == "__main__":
    # Test the `arrange_images` function
    images = [Image.new("RGB", (256, 256), "red") for _ in range(5)]
    combined_image = arrange_images(images, resize_to=512)
    combined_image.show()

    # Test the `display_images` function
    display_images(images)
