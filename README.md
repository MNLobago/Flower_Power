# Transfer Learning with ResNet50 for Custom Image Classification

## Overview

This repository explores transfer learning using the ResNet50 architecture, focusing on fine-tuning for a specific image classification task. The project includes utility functions, model definition, training, and prediction scripts.

## Files

### 1. `README.md`

This markdown document provides an overview of the project, explaining the purpose, approach, and key files.

### 2. `utils.py`

The `utils.py` file contains utility functions for loading and preprocessing data. The `load_data` function prepares the dataset, applying appropriate transformations for training and validation.

### 3. `model.py`

The `model.py` file defines the custom neural network architecture (`CustomModel`) used for transfer learning. It incorporates the ResNet50 architecture with a modified classifier to suit the specific requirements of the image classification task.

### 4. `train.py`

The `train.py` script facilitates the training of the custom model. It includes functions to load data, instantiate the model, define loss functions and optimizers, and run the training loop.

### 5. `predict.py`

The `predict.py` script allows for making predictions using a trained model checkpoint. It includes functions to load a pre-trained model, process an input image, and output predictions with associated probabilities.

## Usage

To train the model and make predictions, follow these steps:

1. **Dataset Preparation**: Ensure your dataset is organized in the required structure (train and valid folders).

2. **Training**: Execute the `train.py` script, specifying the data directory and desired hyperparameters.

    ```bash
    python train.py data_directory --save_dir checkpoint.pth --arch resnet50 --learning_rate 0.001 --hidden_units 1150 228 --epochs 5 --gpu
    ```

3. **Prediction**: After training, use the `predict.py` script to make predictions on new images.

    ```bash
    python predict.py input_image.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
    ```

Feel free to explore and customize the code to fit your specific use case. Enjoy the journey of transfer learning with ResNet50!
