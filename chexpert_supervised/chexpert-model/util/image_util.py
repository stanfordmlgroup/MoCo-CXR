import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.interpolation as interpolate
import torch
from PIL import Image
from PIL import ExifTags

def rotate_img(path):
    import imghdr
    result = imghdr.what(path)
    #print(result)
    img = Image.open(path)
    if result == "jpeg":
        for orientation in ExifTags.TAGS.keys(): 
            if ExifTags.TAGS[orientation]=='Orientation': 
                break 
        
        if img._getexif() is not None:
            exif = dict(img._getexif().items())
            if orientation in exif.keys():
                if exif[orientation] == 3: 
                    img = img.rotate(180, expand=True)
                elif exif[orientation] == 6: 
                    img = img.rotate(270, expand=True)
                elif exif[orientation] == 8: 
                    img = img.rotate(90, expand=True)
    img = img.convert('L').convert('RGB')
    return img

def resize_img(img, scale):
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)
    if max_ind == 0:
        # width fixed at scale
        wpercent = (scale / float(size[0]))
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        # height fixed at scale
        hpercent = (scale / float(size[1]))
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)

    resized_img = cv2.resize(img, desireable_size[::-1])

    return resized_img


def apply_window(img, w_center, w_width, y_min=0., y_max=255., dtype=np.uint8):
    """Window a NumPy array of raw Hounsfield Units.

    Args:
        img: Image to apply the window to. NumPy array of any shape.
        w_center: Center of window.
        w_width: Width of window.
        y_min: Min value for output image.
        y_max: Max value for output image
        dtype: Data type for elements in output image ndarray.

    Returns:
        img_np: NumPy array of after windowing. Values in range [y_min, y_max].
    """
    img_np = np.zeros_like(img, dtype=np.float64)

    # Clip to the lower edge
    x_min = w_center - 0.5 - (w_width - 1.) / 2.
    img_np[img <= x_min] = y_min

    # Clip to the upper edge
    x_max = w_center - 0.5 + (w_width - 1.) / 2.
    img_np[img > x_max] = y_max

    # Scale everything in the middle
    img_np[(img > x_min) & (img <= x_max)] = (((img[(img > x_min) & (img <= x_max)] - (w_center - 0.5))
                                               / (w_width - 1.) + 0.5) * (y_max - y_min) + y_min)

    return img_np.astype(dtype)


def get_crop(bbox):
    """Get crop coordinates and side length given a bounding box.
    Force the crop to be square.

    Args:
        bbox: x1, y1, x2, y2; coordinates for bounding box.

    Returns:
        x1, y1, side_length:
    """
    # Get side length to keep square aspect ratio
    x1, y1, x2, y2 = bbox
    side_length = max(x2 - x1, y2 - y1) + 1

    # Center the skull in the cropped region
    x1 = max(0, x1 - (side_length - (x2 - x1 + 1)) // 2)
    y1 = max(0, y1 - (side_length - (y2 - y1 + 1)) // 2)

    return x1, y1, side_length


def pad_to_shape(array, output_shape, offsets, dtype=np.float32):
    """Pad an array with zeros to the desired output shape.

    Args:
        array: Array to be padded.
        output_shape: The desired shape for the output.
        offsets: List of offsets (will be prepended with zeros
            if fewer dimensions than output_shape).
        dtype: Data type for output array.

    Returns:
        array padded to the given `output_shape`.
    """

    # Create a list of slices from offset to offset + shape in each dimension
    if len(offsets) < len(output_shape):
        offsets = [0] * (len(output_shape) - len(offsets)) + offsets

    insertion_idx = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]

    # Create an array of zeros, may be larger than output shape
    result = np.zeros([max(output_shape[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)],
                    dtype=dtype)

    # Insert the array in the result at the specified offsets
    result[insertion_idx] = array

    # Trim down to output_shape
    result = result[:output_shape[0], :output_shape[1], :output_shape[2]]

    return result


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        assert(len(tensor.size()) == 3), f'Image tensor should have 3 dimensions. Got tensor with {len(tensor.size())} dimenions.'

        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)

        return tensor


def un_normalize(img_tensor, mean, std):
    """Un-normalize a PyTorch Tensor seen by the model into a NumPy array of
    pixels fit for visualization. If using raw Hounsfield Units, window the input.

    Args:
        img_tensor: Normalized tensor using mean and std. Tensor with pixel values in range (-1, 1).

    Returns:
        unnormalized_img: Numpy ndarray with values between 0 and 1.
    """

    unnormalizer = UnNormalize(mean, std)
    # Make a copy, as we don't want to un_normalize in place. The unnormalizer affects the inputted  tensor.
    img_tensor_copy = img_tensor.clone()
    unnormalized_img = unnormalizer(img_tensor_copy)
    unnormalized_img = unnormalized_img.cpu().float().numpy()
    return unnormalized_img


def mask_to_bbox(mask):
    """Convert a mask to bounding box coordinates.

    Args:
        mask: NumPy ndarray of any type, where 0 or false is treated as background.

    Returns:
        x1, y1, x2, y2: Coordinates of corresponding bounding box.
    """
    is_3d = len(mask.shape) == 3

    reduce_axes = (0, 1) if is_3d else (0,)
    cols_any = np.any(mask, axis=reduce_axes)
    cols_where = np.where(cols_any)[0]
    if cols_where.shape[0] == 0:
        return None
    x1, x2 = cols_where[[0, -1]]

    reduce_axes = (0, 2) if is_3d else (1,)
    rows_any = np.any(mask, axis=reduce_axes)
    rows_where = np.where(rows_any)[0]
    if rows_where.shape[0] == 0:
        return None
    y1, y2 = rows_where[[0, -1]]

    return x1, y1, x2, y2


def get_z_range(mask):
    """Get the z-axis range for a mask.

    Args:
        mask: NumPy with mask values, where 0 is treated as background.

    Returns:
        z_range: List with two elements, the min and max z-axis indices containing foreground.
    """
    if len(mask.shape) != 3:
        raise ValueError('Unexpected shape in get_z_range: Needs to be a 3D tensor.')

    z_any = np.any(mask, axis=(1, 2))
    z_where = np.where(z_any)[0]
    if z_where.shape[0] == 0:
        return None
    z_min, z_max = z_where[[0, -1]]

    return [z_min, z_max]


def resize_slice_wise(volume, slice_shape, interpolation_method=cv2.INTER_AREA):
    """Resize a volume slice-by-slice.

    Args:
        volume: Volume to resize.
        slice_shape: Shape for a single slice.
        interpolation_method: Interpolation method to pass to `cv2.resize`.

    Returns:
        Volume after reshaping every slice.
    """
    slices = list(volume)
    for i in range(len(slices)):
        slices[i] = cv2.resize(slices[i], slice_shape, interpolation=interpolation_method)
    return np.array(slices)


def _make_rgb(image):
    """Tile a NumPy array to make sure it has 3 channels."""
    if image.shape[-1] != 3:
        tiling_shape = [1] * (len(image.shape) - 1) + [3]
        return np.tile(image, tiling_shape)
    else:
        return image


def concat_images(images, spacing=10):
    """Concatenate a list of images to form a single row image.

    Args:
        images: Iterable of numpy arrays, each holding an image.
        Must have same height, num_channels, and have dtype np.uint8.
        spacing: Number of pixels between each image.

    Returns: Numpy array. Result of concatenating the images in images into a single row.
    """
    images = [_make_rgb(image) for image in images]
    # Make array of all white pixels with enough space for all concatenated images
    assert spacing >= 0, 'Invalid argument: spacing {} is not non-negative'.format(spacing)
    assert len(images) > 0, 'Invalid argument: images must be non-empty'
    num_rows, _, num_channels = images[0].shape
    assert all([img.shape[0] == num_rows and img.shape[2] == num_channels for img in images]),\
        'Invalid image shapes: images must have same num_channels and height'
    num_cols = sum([img.shape[1] for img in images]) + spacing * (len(images) - 1)
    concatenated_images = np.full((num_rows, num_cols, num_channels), fill_value=255, dtype=np.uint8)

    # Paste each image into position
    col = 0
    for img in images:
        num_cols = img.shape[1]
        concatenated_images[:, col:col + num_cols, :] = img
        col += num_cols + spacing

    return concatenated_images


def stack_videos(img_list):
    """Stacks a sequence of image numpy arrays of shape (num_images x w x h x c) to display side-by-side."""
    # If not RGB, stack to make num_channels consistent
    img_list = [_make_rgb(img) for img in img_list]
    stacked_array = np.concatenate(img_list, axis=2)
    return stacked_array


def add_heat_map(original_image, intensities_np, alpha_img=0.33, color_map='magma', normalize=True):
    """Add a CAM heat map as an overlay on a PNG image.

    Args:
        original_image: Pixels to add the heat map on top of. Must be in range (0, 1).
        intensities_np: Intensity values for the heat map. Must be in range (0, 1).
        alpha_img: Weight for image when summing with heat map. Must be in range (0, 1).
        color_map: Color map scheme to use with PyPlot.
        normalize: If True, normalize the intensities to range exactly from 0 to 1.

    Returns:
        Original pixels with heat map overlaid.
    """
    assert(np.max(intensities_np) <= 1 and np.min(intensities_np) >= 0)
    assert(np.max(original_image) <= 1 and np.min(original_image) >= 0), f'np.max: {np.max(original_image)} and np.min: {np.min(original_image)}'
    color_map_fn = plt.get_cmap(color_map)


    if normalize:
        # Returns pixel values between 0 and 255
        intensities_np = _normalize_png(intensities_np)
    else:
        intensities_np *= 255

    # Get heat map (values between 0 and 1
    heat_map = color_map_fn(intensities_np.astype(np.uint8))
    if len(heat_map.shape) == 3:
        heat_map = heat_map[:, :, :3]
    else:
        heat_map = heat_map[:, :, :, :3]

    new_img = (alpha_img * original_image.astype(np.float32)
            + (1. - alpha_img) * heat_map.astype(np.float32))

    new_img = np.uint8(_normalize_png(new_img))

    return new_img


def dcm_to_png(dcm, w_center=None, w_width=None):
    """Convert a DICOM object to a windowed PNG-format Numpy array.
    Add the given shift to each pixel, clip to the given window, then
    scale pixels to range implied by dtype (e.g., [0, 255] for `uint8`).
    Return ndarray of type `dtype`.

    Args:
        dcm: DICOM object.
        w_center: Window center for windowing conversion.
        w_width: Window width for windowing conversion.

    See Also:
        https://dicom.innolitics.com/ciods/ct-image/voi-lut/00281050
    """
    pixels = dcm.pixel_array
    shift = dcm.RescaleIntercept
    if w_center is None:
        w_center = dcm.WindowCenter
    if w_width is None:
        w_width = dcm.WindowWidth

    img = np.copy(pixels).astype(np.float64) + shift
    img = apply_window(img, w_center, w_width)

    return img


def dcm_to_raw(dcm, dtype=np.int16):
    """Convert a DICOM object to a Numpy array of raw Hounsfield Units.

    Scale by the RescaleSlope, then add the RescaleIntercept (both DICOM header fields).

    Args:
        dcm: DICOM object.
        dtype: Type of elements in output array.

    Returns:
        ndarray of shape (height, width). Pixels are `int16` raw Hounsfield Units.

    See Also:
        https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    """
    img_np = dcm.pixel_array
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    img_np = img_np.astype(dtype)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    img_np[img_np == -2000] = 0

    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope

    if slope != 1:
        img_np = slope * img_np.astype(np.float64)
        img_np = img_np.astype(dtype)

    img_np += int(intercept)
    img_np = img_np.astype(np.int16)

    return img_np


def get_skull_bbox(img):
    """Get a minimal bounding box around the skull.

    Args:
        img: Numpy array of uint8's, after windowing.

    Returns:
        start_x, start_y, end_x, end_y: Coordinates of top-left, bottom-right corners
        for minimal bounding box around the skull.
    """
    _, thresh_img = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    image, contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    skull_bbox = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < 10:
            continue
        x, y, w, h = cv2.boundingRect(c)
        extent = area / float(w * h)
        if extent < 0.2:
            continue
        skull_bbox = get_min_bbox(skull_bbox, (x, y, x + w, y + h))

    return skull_bbox


def get_min_bbox(box_1, box_2):
    """Get the minimal bounding box around two boxes.

    Args:
        box_1: First box of coordinates (x1, y1, x2, y2). May be None.
        box_2: Second box of coordinates (x1, y1, x2, y2). May be None.
    """
    if box_1 is None:
        return box_2
    if box_2 is None:
        return box_1

    b1_x1, b1_y1, b1_x2, b1_y2 = box_1
    b2_x1, b2_y1, b2_x2, b2_y2 = box_2

    x1 = min(b1_x1, b2_x1)
    y1 = min(b1_y1, b2_y1)
    x2 = max(b1_x2, b2_x2)
    y2 = max(b1_y2, b2_y2)

    return x1, y1, x2, y2


def resize(cam, input_img, interpolation='linear'):
    """Resizes a volume using factorized bilinear interpolation"""
    if len(cam.shape) == 2:
        cam = np.expand_dims(cam, axis=0)
    temp_cam = np.zeros((cam.shape[0], input_img.size(2), input_img.size(3)))
    for dim in range(temp_cam.shape[0]):
        temp_cam[dim, :, :] = cv2.resize(cam[dim, :, :], dsize=(temp_cam.shape[1], temp_cam.shape[2]))

    if temp_cam.shape[0] == 1:
        new_cam = np.tile(temp_cam, (input_img.size(1), 1, 1))
    else:
        new_cam = np.zeros((input_img.size(1), temp_cam.shape[1], temp_cam.shape[2]))
        for i in range(temp_cam.shape[1] * temp_cam.shape[2]):
            y = i % temp_cam.shape[2]
            x = (i // temp_cam.shape[2])
            compressed = temp_cam[:, x, y]
            labels = np.arange(compressed.shape[0], step=1)
            new_labels = np.linspace(0, compressed.shape[0] - 1, new_cam.shape[0])
            f = interpolate.interp1d(labels, compressed, kind=interpolation)
            expanded = f(new_labels)
            new_cam[:, x, y] = expanded

    return new_cam


def _normalize_png(img):
    """Normalizes img to be in the range 0-255."""
    img -= np.amin(img)
    img /= (np.amax(img) + 1e-7)
    img *= 255
    return img

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
