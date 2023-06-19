# Airbus Ship Detection Solution

This repository contains a solution for the Airbus Ship Detection problem using a U-Net convolutional neural network. The solution includes separate files for data processing, model training, and model inference.

## Problem Description

The goal of the Airbus Ship Detection problem is to develop a model that can detect ships in satellite images. Given a set of images and corresponding ship masks, the task is to train a model that can accurately identify the presence of ships in new images.

## Solution Overview

The solution is implemented using Python and TensorFlow. It consists of the following files:

- `data.py`: This file contains functions for processing the input data and creating the dataset for training and testing the model.
- `train.py`: This file is used to train the U-Net model on the prepared dataset. It loads the training data, creates the model architecture, compiles the model, trains it on the data, evaluates its performance, and saves the trained model.
- `inference.py`: This file is used to load a trained model and perform inference on new images. It takes an input image, preprocesses it, applies the trained model to generate ship predictions, and visualizes the results.

Additionally, there are two Jupyter Notebook files:

- `test-task-Winstars.ipynb`: This notebook provides exploratory data analysis (EDA) on the Airbus Ship Detection dataset. It includes data visualization, statistical analysis, and insights into the characteristics of the images and ship masks. It also includes model creation and training.
- `test-task-Winstars-ovp.ipynb.ipynb: This notebook demonstrates the model training process using the old package versions, which is faster and allows you to evaluate the final result.

## Usage

1. Install the required dependencies listed in the `requirements.txt` file.
2. Prepare the dataset by placing the training images in the specified directory and creating the ship masks CSV file.
3. Modify the necessary paths and parameters in the `data.py`, `train.py`, and `inference.py` files to match your setup.
4. Run the `train.py` file to train the U-Net model. It will output the model's training progress and evaluate its performance on the test data.
5. Once trained, you can use the `inference.py` file to perform ship detection on new images. Provide the path to the input image, and it will generate the predicted ship masks and visualize the results.
