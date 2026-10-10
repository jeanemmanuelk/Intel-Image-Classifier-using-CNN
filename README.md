# Image Classifier using Convolutional Neural Network (CNN)

## Overview

This project implements an image classifier using a Convolutional Neural Network (CNN) to classify images into six categories: mountain, street, glacier, buildings, sea, and forest. The full challenge dataset has about 25,000 images across the six categories. This notebook uses the 17,034 labeled images (14,034 for training, plus 3,000 split evenly into 1,500 validation and 1,500 test), resized to 150x150. The goal of this project is to build a CNN model that can accurately classify images into their respective categories.

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

The trained CNN reaches a validation accuracy of about 80% (0.80 after 20 epochs), while its training accuracy climbs to about 97%. That gap, and a validation loss that bottoms out around epoch 4 (about 0.56) and climbs to about 1.2 by epoch 20, mean the model overfits. Adding data augmentation, early stopping, or a pretrained backbone are the obvious next things to try. The notebook also computes accuracy, precision, recall and F1 on the test set, but does not print them. Only a confusion matrix plot and a few sample predictions are shown.

## Conclusion

This image classification project demonstrates how to build a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images into different categories. The model achieved a reasonable accuracy on the validation data and can be further fine-tuned and optimized to improve performance. By following the steps outlined in this notebook, users can easily apply CNNs to their own image classification tasks and explore various techniques for improving model accuracy and robustness.
