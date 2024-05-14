# CPSC542 Extra Credit 1
### Student Information
Student Name: Devyn Miller
Student ID: 2409539
Collaboration: None
### Resources
Resources used:
*Due to the large size of the Keras pretrained models and the image datasets used, they are not included in the repository. They can be accessed through the following links:*
- **Keras Pretrained Models**: [Kaggle - Keras Pretrained Models](https://www.kaggle.com/datasets/gaborfodor/keras-pretrained-models)
- **Fire Dataset**: [Kaggle - Fire Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
- Perplexity
- Kaggle (delllectron)


Code Repository
[GitHub Repository for CPSC542 Extra Credit 1](https://github.com/devyn-miller/cpsc542-extra-credit)

# Project Overview

This project involves the classification of images into 'fire' and 'non-fire' categories using machine learning models. Below is a description of each file in the project and their respective roles:

## Files Description
  `main.ipynb`
  This Jupyter notebook serves as the main file where various data processing, model training, and visualization tasks are performed. Key functionalities include:
  Image Loading and Preprocessing: Images are loaded and preprocessed to fit the model's input requirements.
  Model Training: A machine learning model is trained using the preprocessed images.
  Visualization: Various visualizations such as displaying images, plotting Grad-CAM heatmaps, and showing model predictions.
  Exploratory Data Analysis (EDA): Basic EDA tasks to understand the dataset better.
  
  
  `3b3w.py`
  A Python script that contains utility functions and model definitions. Key components include:
  Model Definition: Contains the definition of the neural network model used for predictions.
  Image Preprocessing: Functions to load and preprocess images for model prediction.
  Utility Functions: Includes functions to display images and other utility tasks.
  
  
  `preprocessing.py`
  This script includes functions related to image preprocessing such as:
  Reading Images: Functions to read images from disk.
  Image Segmentation: Methods to segment images based on certain features.
  Image Sharpening: Functions to enhance the sharpness of images.
  
  
  `augmentation.py`
  Contains functions for augmenting images to create a more robust dataset. This includes:
  Data Augmentation: Techniques like rotation, zoom, and flip to augment the images.
  
  
  `model.py`
  Defines the structure of the neural network models used for training and prediction. It includes:
  Model Architecture: Detailed layer-by-layer architecture of the model.
  Compilation and Training: Functions to compile and train the model.
  
  
  `predict.py`
  This script is responsible for making predictions using trained models. It includes:
  Load Model: Function to load a pre-trained model.
  Predict: Function to make predictions on new data.


  `validate.py`
  Contains functions for model validation and evaluation. Key functionalities include:
  Classification Report: Generates a classification report for model evaluation.
  Confusion Matrix: Function to plot a confusion matrix.
