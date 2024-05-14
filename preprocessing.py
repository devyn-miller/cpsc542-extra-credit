import os
import cv2
import numpy as np
from keras.preprocessing import image

def create_mask_for_plant(image):
    """
    Create a binary mask for the plant in an image based on HSV color space thresholding.

    Parameters:
    image (numpy.ndarray): The input image in BGR color space.

    Returns:
    numpy.ndarray: A binary mask where white represents the plant and black represents the background.
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0,0,250])
    upper_hsv = np.array([250,255,255])
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def segment_image(image):
    """
    Segment the plant from the background in an image.

    Parameters:
    image (numpy.ndarray): The input image in BGR color space.

    Returns:
    numpy.ndarray: The segmented image with the plant isolated from the background.
    """
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output/255

def sharpen_image(image):
    """
    Sharpen the input image using Gaussian blur and weighted addition.

    Parameters:
    image (numpy.ndarray): The input image in BGR color space.

    Returns:
    numpy.ndarray: The sharpened image.
    """
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

def read_img(filepath, size):
    """
    Load an image from a specified file path and resize it to a target size.

    Parameters:
    filepath (str): The path to the image file.
    size (tuple): The target size to resize the image to.

    Returns:
    numpy.ndarray: The loaded image as a numpy array.
    """
    img = image.load_img(os.path.join('/Users/devynmiller/Downloads/ec may 14 cpsc/input/fire_dataset', filepath), target_size=size)
    img = image.img_to_array(img)
    return img
