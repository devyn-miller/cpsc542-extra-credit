from scipy.ndimage import rotate

def augment_data(data):
    """
    Augment the input image by rotating it.

    This function rotates the input image data by 45 degrees without changing the shape of the image.

    Parameters:
    data (ndarray): The image data to be augmented.

    Returns:
    ndarray: The rotated image data.
    """
    return rotate(data, 45, reshape=False)
