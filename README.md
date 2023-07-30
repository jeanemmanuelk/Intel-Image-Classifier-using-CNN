# Image Classifier using Convolutional Neural Network (CNN)

## Overview

This project implements an image classifier using a Convolutional Neural Network (CNN) to classify images into six categories: mountain, street, glacier, buildings, sea, and forest. The dataset contains approximately 25,000 images of size 150x150 distributed across the six categories. The goal of this project is to build a CNN model that can accurately classify images into their respective categories.

## Data

The dataset used for this project was initially published on https://datahack.analyticsvidhya.com by Intel for an Image Classification Challenge. The dataset is split into training and testing sets, and it is organized into folders where each category has its own subfolder containing images.

## Steps

1. **Importing Libraries:** Necessary Python libraries are imported, including TensorFlow for deep learning, OpenCV for image processing, NumPy for numerical operations, and Matplotlib for visualization. Some specific UserWarnings are filtered to improve code readability.

2. **Data Processing:** The data is loaded and preprocessed using the `load_images` function. Images are resized to a common size of 150x150 pixels, and the pixel values are normalized to range between 0 and 1. The data is then split into training, validation, and testing sets.

3. **Data Exploration and Visualization:** The shapes of the training, validation, and testing sets are displayed to understand the number of images in each set. Functions for plotting random images and a grid of random images are defined to visualize sample images from the dataset.

4. **Model Building (CNN):** A Convolutional Neural Network (CNN) is constructed using TensorFlow's Keras API. The model consists of two convolutional layers with MaxPooling, followed by a few dense layers and a final output layer with six units (equal to the number of categories). The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy as the evaluation metric.

5. **Model Training:** The model is trained on the training data using the `fit` function with a batch size of 32 and for 20 epochs. The training and validation accuracy and loss are recorded during the training process.

6. **Model Evaluation and Prediction:** The trained model is evaluated on the validation data, and the accuracy and loss are printed. The model's predictions are made on the test data, and evaluation metrics such as accuracy, precision, recall, and F1-score are calculated. The confusion matrix is visualized to understand the model's performance on different categories.

7. **Model Performance Plots:** Plots for training and validation accuracy and loss are generated to visualize the model's learning progress during training.

## Results

The trained CNN model achieves a validation accuracy of approximately 80%. The model's performance can be further assessed using the confusion matrix and sample image predictions.

## Conclusion

This image classification project demonstrates how to build a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images into different categories. The model achieved a reasonable accuracy on the validation data and can be further fine-tuned and optimized to improve performance. By following the steps outlined in this notebook, users can easily apply CNNs to their own image classification tasks and explore various techniques for improving model accuracy and robustness.
