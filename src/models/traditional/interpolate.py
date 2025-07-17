import numpy as np
from PIL import Image
from scipy.interpolate import griddata


def interpolate_sparse_image(image):
    img_array = np.array(image)
    height, width, _ = img_array.shape

    # create coordinate grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # find the position of all opaque pixels (pixels with alpha channel > 0)
    valid_mask = img_array[:, :, 3] > 0
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # if there are no valid pixels, return the original image
    if len(x_valid) == 0:
        return image

    # prepare the coordinates for interpolation
    points = np.column_stack((x_valid, y_valid))

    # create a complete target grid point
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    targets = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # interpolate the three channels of RGB respectively
    result_array = np.zeros((height, width, 4), dtype=np.uint8)
    for channel in range(3):  # RGB channels
        values = img_array[:, :, channel][valid_mask]

        # interpolate using griddata
        interpolated = griddata(
            points, values, targets, method="linear", fill_value=np.mean(values)
        )

        result_array[:, :, channel] = interpolated.reshape(height, width)

    # set the alpha channel to be completely opaque
    result_array[:, :, 3] = 255

    # keep the values of the original opaque pixels unchanged
    result_array[valid_mask] = img_array[valid_mask]

    # RGBA -> RGB
    result_array = result_array[:, :, :3]

    return Image.fromarray(result_array)
