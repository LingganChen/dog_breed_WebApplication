# DogBreed Classification Project

## Overview

This project utilizes TensorFlow and Keras with the InceptionV3 model to create a neural network for classifying over 70 different dog breeds based on images. The primary objective is to develop an accurate and robust model that can correctly identify the breed of a dog from an input image.

## Table of Contents

- [Project Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
## Installation

To run this project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/LingganChen/dog_breed_WebApplication.git
    cd dog_breed_WebApplication
    ```
2. Set up environment py -m venv .env
3. Activate environment .env\scripts\activate   
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have the required dataset.
2. Set up environment in cmd: py -m venv .env
3. Activate environment .env\scripts\activate  
4. Run application using- flask run
5. Open the local html link in cmd
6. Once you are on the web page, select an image and click on submit to see the prediction
   
## Dataset

The dataset used for this project includes images of over 70 different dog breeds. It is sourced from Kaggle(https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set/data), and it is preprocessed to be compatible with the InceptionV3 model.

## Model Architecture

The neural network architecture is based on the InceptionV3 model, a powerful convolutional neural network (CNN) known for its effectiveness in image classification tasks.

## Training

The model is trained using the prepared dataset, and the training process is configured to optimize accuracy and minimize loss. Training progress and performance metrics are logged for analysis.

## Results

The final model achieves 93% accuracy, demonstrating its effectiveness in classifying dog breeds from images.
