# Age Prediction from Facial Images

This project is focused on predicting the age of individuals from facial images. It uses a deep learning model based on the ResNet50 architecture, pre-trained on ImageNet, which is fine-tuned on the UTKFace dataset. The goal is to predict a person's age as accurately as possible based on their facial features.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Overview
The model uses Convolutional Neural Networks (CNNs) to estimate a person's age based on their facial image. The ResNet50 model serves as the backbone of the architecture, which is fine-tuned on the UTKFace dataset for age prediction. The project aims to provide a robust solution for age estimation from images.

## Dependencies
To run this project, you will need the following Python libraries:

- `tensorflow` >= 2.5
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `keras`
- `opencv-python`

You can install the necessary libraries by running:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib keras opencv-python

