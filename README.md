# DigitRecognition

## Overview

This project implements a digit recognition system using the MNIST dataset and a neural network model built with TensorFlow and Keras. The trained model can predict handwritten digits from images provided by users.

## Requirements

- Python 3.12.2
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

## Installation

**Clone the repository:**

    git clone https://github.com/Sumeet-2023/DigitRecognition.git

## Usage

1. **Training the Model:**
   
   Before using the digit recognition system, you may want to train the model. If you haven't already trained it, uncomment the training code in the script and run it. This step involves loading the MNIST dataset, preprocessing the data, building the neural network model, training it on the training data, and saving the trained model.

    ```bash
    python digit_recognition.py
    ```

2. **Using the Trained Model:**

   To use the trained model for digit recognition, ensure you have saved the model as 'DigitRecognition.keras' in the project directory. Then, provide the images containing handwritten digits in the 'Digits' directory. The script will read each image, preprocess it, make predictions using the trained model, and display the predicted digit along with the image.

    ```bash
    python digit_recognition.py
    ```

## File Structure

- `digit_recognition.py`: Main script implementing the digit recognition system.
- `Digits/`: Directory containing images of handwritten digits for testing.
- `mnist.npz`: MNIST dataset file.
