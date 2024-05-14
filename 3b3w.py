from model import build_model
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
import os

def load_and_preprocess_image(img_path, target_size=(255, 255)):
    """
    Load an image from the specified path and preprocess it for model prediction.

    This function performs the following operations:
    1. Loads an image from the specified file path.
    2. Resizes the image to the specified target size.
    3. Converts the image to a numpy array.
    4. Adds a batch dimension to the array.
    5. Preprocesses the array using the Xception model's preprocessing function.

    Parameters:
    img_path (str): The file path to the image.
    target_size (tuple): The target size to which the image is resized, given as (height, width).

    Returns:
    ndarray: The preprocessed image array, ready for input into a model.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


# Function to display images
def display_images(indices, title, image_paths):
    """
    Display a set of images specified by their indices in a list of image paths.

    This function performs the following operations:
    1. Creates a new figure with a specified size.
    2. Iterates over the provided indices to load and display each image.
    3. Sets the title for each image based on the provided title and the image's position.
    4. Turns off the axis for each image plot.
    5. Displays the plot with all the images.

    Parameters:
    indices (list of int): A list of indices specifying which images to display from the image_paths list.
    title (str): The base title to use for each image.
    image_paths (list of str): A list containing the paths to the images.

    Returns:
    None
    """
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(indices, 1):
        img = image.load_img(image_paths[idx], target_size=(255, 255))
        plt.imshow(img)
        plt.title(f"{title} {i}")
        plt.axis('off')
    plt.show()