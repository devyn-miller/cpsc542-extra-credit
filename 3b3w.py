from model import build_model
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
import os

# Function to load and preprocess images
def load_and_preprocess_image(img_path, target_size=(255, 255)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


# Function to display images
def display_images(indices, title):
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(indices, 1):
        img = image.load_img(image_paths[idx], target_size=(255, 255))
        plt.imshow(img)
        plt.title(f"{title} {i}")
        plt.axis('off')
    plt.show()