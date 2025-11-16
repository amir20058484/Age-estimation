# Age Prediction from Images using Deep Learning

This project focuses on predicting the age of individuals based on their facial images using deep learning techniques. The model utilizes a pre-trained ResNet50 architecture, which is fine-tuned on the UTKFace dataset. The goal is to predict a continuous age value from images.

## Table of Contents
1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Results](#results)
7. [Usage](#usage)
8. [License](#license)

## Introduction
This project leverages Convolutional Neural Networks (CNNs) for facial age estimation. The model is based on the ResNet50 architecture, which is a state-of-the-art deep learning model, pre-trained on ImageNet. We fine-tune this model on the UTKFace dataset, a large dataset containing images of faces with age labels.

## Dependencies
The following Python libraries are required to run this project:

- `tensorflow` >= 2.5
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `keras`
- `opencv-python`

You can install the necessary dependencies by running the following command:

```bash
pip install -r requirements.txt
