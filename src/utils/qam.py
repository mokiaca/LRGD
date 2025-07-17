"""
This file is modified from https://github.com/ispamm/Img2Img-SC/blob/main/scripts/qam.py#L125
"""

import math
import random
from typing import List, Tuple

import bitstring
import numpy as np
import torch
from PIL import Image
from scipy.special import erfc


def qam16ModulationTensor(
    input_tensor: torch.Tensor, snr_db: float = 10
) -> torch.Tensor:
    """
    Modulate Tensor in 16QAM transmission and simulate noisy channel conditions.

    Parameters:
        input_tensor (torch.Tensor): Input tensor to be modulated
        snr_db (float): Signal-to-Noise ratio in dB

    Returns:
        torch.Tensor: Modulated and noisy tensor with same shape as input
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    if not input_tensor.is_floating_point():
        input_tensor = input_tensor.float()

    message_shape = input_tensor.shape
    original_device = input_tensor.device
    try:
        bit_list = tensor2bin(input_tensor)
        bit_list_noisy = introduce_noise(bit_list, snr=snr_db)
        back_to_tensor = bin2tensor(bit_list_noisy)
        result = back_to_tensor.reshape(message_shape).to(original_device)
        return result
    except Exception as e:
        raise RuntimeError(f"Error during QAM modulation: {str(e)}")


def qam16ModulationString(input_string: str, snr_db: float = 10) -> str:
    """
    Modulate String in 16QAM transmission and simulate noisy channel conditions.

    Parameters:
        input_string (str): Input string to be modulated
        snr_db (float): Signal-to-Noise ratio in dB

    Returns:
        str: Modulated and noisy string
    """
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")

    if not input_string:
        return ""

    try:
        bit_list = list2bin(input_string)
        bit_list_noisy = introduce_noise(bit_list, snr=snr_db)
        back_to_chars = bin2list(bit_list_noisy)
        valid_chars = [char for char in back_to_chars if char is not None]
        return "".join(valid_chars)
    except Exception as e:
        raise RuntimeError(f"Error during QAM string modulation: {str(e)}")


def qam16ModulationImage(input_image: Image.Image, snr_db: float = 10) -> Image.Image:
    """
    Modulate PIL Image in 16QAM transmission and simulate noisy channel conditions.

    Parameters:
        input_image (PIL.Image): Input image to be modulated
        snr_db (float): Signal-to-Noise ratio in dB

    Returns:
        PIL.Image: Modulated and noisy image
    """
    if not isinstance(input_image, Image.Image):
        raise TypeError("Input must be a PIL.Image")

    try:
        original_mode = input_image.mode
        img_array = np.array(input_image)
        original_shape = img_array.shape
        is_binary = is_binary_image(img_array) and img_array.max() <= 1

        img_tensor = torch.from_numpy(img_array).float()

        if is_binary:
            bit_list = binary_image_to_bin(img_tensor)
            bit_list_noisy = introduce_noise(bit_list, snr=snr_db)  # ------> Split
            noisy_tensor = binary_bin_to_tensor(bit_list_noisy, original_shape)
            noisy_array = noisy_tensor.numpy()
            noisy_image = Image.fromarray(noisy_array.astype(np.uint8) * 255)
        else:
            img_tensor = (img_tensor / 127.5) - 1.0  # Normalize to [-1, 1]
            bit_list = image_tensor2bin(img_tensor)
            bit_list_noisy = introduce_noise(bit_list, snr=snr_db)  # ------> Split
            noisy_tensor = image_bin2tensor(bit_list_noisy, original_shape)
            noisy_tensor = ((noisy_tensor + 1.0) * 127.5).clamp(0, 255)
            noisy_array = noisy_tensor.numpy().astype(np.uint8)

            # MARK: for RGBA image, keep the transparent pixels transparent
            if noisy_array.ndim == 3 and noisy_array.shape[2] == 4:
                noisy_array[:, :, 3] = np.where(img_array[:, :, 3] == 0, 0, 255)

            noisy_image = Image.fromarray(noisy_array, mode=original_mode)

        return noisy_image

    except Exception as e:
        raise RuntimeError(f"Error during image QAM modulation: {str(e)}")


def introduce_noise(bit_list, snr=10, qam=16):
    """
    Introduce noise to a bit sequence based on QAM modulation

    Parameters:
        bit_list: List of bit strings
        snr: Signal-to-Noise ratio in dB
        qam: QAM modulation order (must be a perfect square)

    Returns:
        List of bit strings with noise-induced errors
    """
    # Validate inputs
    if not np.sqrt(qam).is_integer():
        raise ValueError("QAM order must be a perfect square")

    # Compute ebno according to SNR
    ebno = 10 ** (snr / 10)

    # Calculate bit error probability for QAM
    K = np.sqrt(qam)
    M = 2**K

    # Symbol error probability for QAM
    Pm = (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 * K * ebno / (2 * (M - 1))))

    # Symbol to bit error conversion
    Ps_qam = 1 - (1 - Pm) ** 2  # Symbol error probability
    Pb_qam = Ps_qam / K  # Bit error probability

    bit_flipped = 0
    bit_tot = 0
    new_list = []

    # Apply noise by randomly flipping bits
    for num in bit_list:
        num_new = []
        for b in num:
            if random.random() < Pb_qam:
                num_new.append(str(1 - int(b)))  # Flip the bit
                bit_flipped += 1
            else:
                num_new.append(b)
            bit_tot += 1
        new_list.append("".join(num_new))

    # Print bit error rate for debugging
    # if bit_tot > 0:
    #     ber = bit_flipped / bit_tot
    #     print(f"Actual Bit Error Rate: {ber:.6f}")
    #     print(f"Theoretical Bit Error Rate: {Pb_qam:.6f}")

    return new_list


def bin2float(b):
    """Convert binary string to a float.

    Attributes:
        :b: Binary string to transform.
    """

    num = bitstring.BitArray(bin=b).float
    # Adjust for NaN, Inf, and extreme values
    if (
        math.isnan(num)
        or math.isinf(num)
        or num > 10  # Sometimes need to be adjusted
        or num < -10
        or -1e-2 < num < 1e-2
    ):
        num = np.random.randn()
    return num


def float2bin(f: float) -> str:
    """Convert float to 64-bit binary string with error handling."""
    try:
        f1 = bitstring.BitArray(float=float(f), length=64)
        return f1.bin
    except Exception as e:
        raise ValueError(f"Could not convert {f} to binary: {str(e)}")


def tensor2bin(tensor: torch.Tensor) -> List[str]:
    """Convert torch.Tensor to list of binary strings."""
    try:
        tensor_flattened = tensor.cpu().detach().reshape(-1).numpy()
        return [float2bin(number) for number in tensor_flattened]
    except Exception as e:
        raise ValueError(f"Error converting tensor to binary: {str(e)}")


def bin2tensor(input_list: List[str]) -> torch.Tensor:
    """Convert list of binary strings to torch.Tensor."""
    try:
        tensor_reconstructed = [bin2float(b) for b in input_list]
        return torch.tensor(tensor_reconstructed, dtype=torch.float32)
    except Exception as e:
        raise ValueError(f"Error converting binary to tensor: {str(e)}")


def string2int(char: str):
    return ord(char)


def int2bin(int_num: int):
    return "{0:08b}".format(int_num)


def int2string(int_num: int):
    return chr(int_num)


def bin2int(bin_num):
    return int(bin_num, 2)


def list2bin(input_list: list):
    return [int2bin(string2int(number)) for number in input_list]


def bin2list(input_list):
    return [int2string(bin2int(bin)) for bin in input_list]


def image_tensor2bin(tensor: torch.Tensor) -> List[str]:
    """
    Convert image tensor to list of binary strings.
    Each pixel value is converted to an 8-bit binary string.
    """
    try:
        # Flatten tensor and convert to numpy
        tensor_flat = tensor.cpu().flatten().numpy()

        # Convert each value to binary string
        bit_list = []
        for val in tensor_flat:
            # Convert float to 8-bit integer representation
            int_val = int((val + 1.0) * 127.5)
            int_val = max(0, min(255, int_val))  # Clamp to valid range
            bin_str = format(int_val, "08b")  # 8-bit binary string
            bit_list.append(bin_str)

        return bit_list

    except Exception as e:
        raise ValueError(f"Error converting image tensor to binary: {str(e)}")


def image_bin2tensor(bit_list: List[str], original_shape: Tuple) -> torch.Tensor:
    """
    Convert list of binary strings back to image tensor.
    Each 8-bit binary string is converted back to a pixel value.
    """
    try:
        # Convert binary strings back to integer values
        values = []
        for bin_str in bit_list:
            try:
                int_val = int(bin_str, 2)
                values.append(int_val)
            except ValueError:
                # Handle corrupted binary strings
                values.append(random.randint(0, 255))

        # Convert to tensor and reshape
        tensor = torch.tensor(values, dtype=torch.float32)
        tensor = tensor.reshape(original_shape)

        # Normalize back to [-1, 1] range
        tensor = (tensor / 127.5) - 1.0

        return tensor

    except Exception as e:
        raise ValueError(f"Error converting binary to image tensor: {str(e)}")


def is_binary_image(img: np.ndarray | torch.Tensor | Image.Image) -> bool:
    """
    Check if an image is binary (black and white).
    """
    if isinstance(img, Image.Image):
        img = np.array(input).astype(np.float32)

    unique_values = np.unique(img)
    return (
        len(unique_values) <= 2
        and 0 in unique_values
        and (1 in unique_values or 255 in unique_values)
    )


def binary_image_to_bin(tensor: torch.Tensor) -> List[str]:
    """
    Convert binary image tensor to list of binary strings.
    Each pixel is converted to a single bit.

    Parameters:
        tensor: Input tensor with values 0/1 or 0/255

    Returns:
        List of single-bit strings ('0' or '1')
    """
    try:
        # Normalize values to 0/1
        if tensor.max() > 1:
            tensor = (tensor > 127).float()

        # Flatten tensor and convert to binary strings
        tensor_flat = tensor.cpu().flatten().bool()
        return ["1" if px else "0" for px in tensor_flat]

    except Exception as e:
        raise ValueError(f"Error converting binary image to bits: {str(e)}")


def binary_bin_to_tensor(bit_list: List[str], original_shape: Tuple) -> torch.Tensor:
    """
    Convert list of binary strings back to binary image tensor.
    Each bit string represents a single pixel.

    Parameters:
        bit_list: List of single-bit strings ('0' or '1')
        original_shape: Shape of the original image

    Returns:
        Binary tensor with values 0 or 1
    """
    try:
        # Convert binary strings to values
        values = []
        for bit in bit_list:
            try:
                val = int(bit, 2)
                values.append(val)
            except ValueError:
                # Handle corrupted bits by random assignment
                values.append(random.randint(0, 1))

        # Convert to tensor and reshape
        tensor = torch.tensor(values, dtype=torch.bool)
        return tensor.reshape(original_shape)

    except Exception as e:
        raise ValueError(f"Error converting bits to binary image: {str(e)}")


if __name__ == "__main__":
    import requests

    snr_db = 1

    # Test QAM modulation with random tensor
    tensor = torch.randn(3, 256, 256)
    noisy_tensor = qam16ModulationTensor(tensor, snr_db=snr_db)

    # Test QAM modulation with random string
    input_string = "Hello, World!" * 10
    noisy_string = qam16ModulationString(input_string, snr_db=snr_db)
    print(noisy_string)

    # Test QAM modulation with random image
    # input_image = Image.new("RGB", (256, 256))
    url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    input_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    noisy_image = qam16ModulationImage(input_image, snr_db=snr_db)
    noisy_image.show()
    noisy_image.save("noisy_image.png")
    print("Noisy image saved to 'noisy_image.png'")
