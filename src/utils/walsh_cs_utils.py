"""
This module contains utility functions for Walsh-Hadamard Compressed Sensing.
"""

import math
from typing import Tuple

import numpy as np
from loguru import logger
from PIL import Image


def walsh_hadamard_matrix(n: int) -> np.ndarray:
    """Generate Walsh-Hadamard matrix of size n x n. n must be a power of 2."""
    if n <= 0 or (n & (n - 1)) != 0:  # Check if n is power of 2
        raise ValueError("Size must be a power of 2")

    H = np.array([[1]])
    while H.shape[0] < n:
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))
    return H / np.sqrt(n)  # Normalize


def pad_to_power_of_two(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad image to nearest power of 2 in both dimensions."""
    h, w = image.shape[:2]
    new_h = 2 ** math.ceil(math.log2(h))
    new_w = 2 ** math.ceil(math.log2(w))

    padded = np.zeros((new_h, new_w, *image.shape[2:]))
    padded[:h, :w] = image
    return padded, (h, w)


def block_process(image: np.ndarray, block_size: int = 32) -> np.ndarray:
    """Split image into blocks of size block_size x block_size."""
    h, w = image.shape[:2]
    blocks_h = h // block_size
    blocks_w = w // block_size

    blocks = []
    for i in range(blocks_h):
        for j in range(blocks_w):
            block = image[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            blocks.append(block)
    return np.array(blocks)


def walsh_hadamard_encode(
    image: Image.Image, sampling_ratio: float, block_size: int = None
) -> Tuple[np.ndarray, dict]:
    """
    Perform Walsh-Hadamard compression sampling on the image.

    Args:
        image: input image
        sampling_ratio: sampling rate (0-1)
        block_size: image block size

    Returns:
        Tuple[np.ndarray, dict]:
            - compressed data array, shape is (number of blocks, number of samples, number of channels)
            - metadata dictionary required for reconstruction
    """
    if block_size is None:
        _h, _w = image.size
        block_size = 2 ** max(math.floor(math.log2((_h * _w) ** 0.5) / 2), 2)

    # Convert to numpy array
    img_array = np.array(image)

    # Padding
    padded_img, original_size = pad_to_power_of_two(img_array)

    # Generate Walsh-Hadamard sampling matrix
    n = block_size * block_size
    H = walsh_hadamard_matrix(n)
    m = int(n * sampling_ratio)
    if m == 0:
        m = 1
        logger.warning("Sampling ratio is too low, setting to 1")
    H_sampled = H[:m, :]

    # Get image information
    h, w = padded_img.shape[:2]
    blocks_h = h // block_size
    blocks_w = w // block_size
    n_channels = padded_img.shape[2] if len(padded_img.shape) > 2 else 1

    # Store compressed data
    compressed_data = np.zeros((blocks_h * blocks_w, m, n_channels))

    # Process each channel
    for c in range(n_channels):
        img_channel = padded_img[..., c] if n_channels > 1 else padded_img
        blocks = block_process(img_channel, block_size)

        # Compress and sample each block
        for i, block in enumerate(blocks):
            flat_block = block.reshape(-1)
            compressed_data[i, :, c if n_channels > 1 else 0] = np.dot(
                H_sampled, flat_block
            )

    # Save metadata for reconstruction
    metadata = {
        "original_size": original_size,
        "padded_shape": padded_img.shape,
        "block_size": block_size,
        "sampling_ratio": sampling_ratio,
        "n_channels": n_channels,
        "blocks_h": blocks_h,
        "blocks_w": blocks_w,
    }
    # for better CR
    compressed_data = compressed_data.astype(np.float16)

    return compressed_data, metadata


def walsh_hadamard_decode(compressed_data: np.ndarray, metadata: dict) -> Image.Image:
    """
    Reconstruct the image from compressed data.

    Args:
        compressed_data: compressed data, shape is (number of blocks, number of samples, number of channels)
        metadata: metadata dictionary required for reconstruction

    Returns:
        Image.Image: reconstructed image
    """
    # Decode metadata
    block_size = metadata["block_size"]
    sampling_ratio = metadata["sampling_ratio"]
    n_channels = metadata["n_channels"]
    blocks_h = metadata["blocks_h"]
    blocks_w = metadata["blocks_w"]
    original_size = metadata["original_size"]
    padded_shape = metadata["padded_shape"]

    # Reconstruct Walsh-Hadamard matrix
    n = block_size * block_size
    H = walsh_hadamard_matrix(n)
    m = int(n * sampling_ratio)
    if m == 0:
        m = 1
    H_sampled = H[:m, :]

    # Create output image array
    reconstructed = np.zeros(padded_shape)

    # Reconstruct each channel
    for c in range(n_channels):
        reconstructed_channel = np.zeros((padded_shape[0], padded_shape[1]))

        # Reconstruct each block
        for idx in range(blocks_h * blocks_w):
            # Get compressed data for current block
            sampled = compressed_data[idx, :, c if n_channels > 1 else 0]

            # Reconstruct block
            reconstructed_block = np.dot(H_sampled.T, sampled)
            reconstructed_block = reconstructed_block.reshape(block_size, block_size)

            # Put reconstructed block back to correct position
            i = idx // blocks_w
            j = idx % blocks_w
            reconstructed_channel[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ] = reconstructed_block

        # Put reconstructed channel into result array
        if n_channels > 1:
            reconstructed[..., c] = reconstructed_channel
        else:
            reconstructed = reconstructed_channel

    # Crop back to original size
    reconstructed = reconstructed[: original_size[0], : original_size[1]]
    if n_channels > 1:
        reconstructed = reconstructed[:, :, :n_channels]

    return Image.fromarray(np.uint8(np.clip(reconstructed, 0, 255)))


if __name__ == "__main__":
    image = Image.open("test.jpg")
    compressed_data, metadata = walsh_hadamard_encode(image, 0.01)

    original_size = metadata["original_size"]
    compression_ratio = 1 - (
        compressed_data.size
        / (original_size[0] * original_size[1] * metadata["n_channels"])
    )
    print(f"Compressing Ratio: {compression_ratio:.2%}")

    reconstructed = walsh_hadamard_decode(compressed_data, metadata)
    # reconstructed.show()
    reconstructed.save("reconstructed.jpg")
