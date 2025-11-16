{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Flower Classification Model using ResNet50\n",
        "\n",
        "This notebook demonstrates training and fine-tuning a deep learning model to classify 17 different flower species using transfer learning with ResNet50. The model is built with TensorFlow and Keras, incorporating data augmentation, training on a dataset, fine-tuning, and inference on sample images.\n",
        "\n",
        "## Project Overview\n",
        "\n",
        "- **Task**: Multi-class image classification for 17 flower categories.\n",
        "- **Model Architecture**: Pre-trained ResNet50 (from ImageNet) as the base, with additional layers for classification.\n",
        "- **Dataset**: The [17 Category Flower Dataset](https://www.kaggle.com/datasets/rajatkumar30/flower-recognition) from Kaggle, consisting of training and test images.\n",
        "- **Key Features**:\n",
        "  - Data augmentation for improved generalization.\n",
        "  - Initial training with frozen base layers.\n",
        "  - Fine-tuning of the last 10 layers for better performance.\n",
        "  - Visualization of training and validation accuracy.\n",
        "  - Sample prediction on a new image.\n",
        "\n",
        "This code is designed for a Kaggle notebook environment but can be adapted for local runs with minor modifications (e.g., dataset paths)."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Dataset\n",
        "\n",
        "The dataset includes:\n",
        "- **Training Set**: Images in `/train` directory, organized by class folders.\n",
        "- **Test Set**: Images in `/test` directory.\n",
        "- **Classes**: 17 flower types (e.g., Bluebell, Buttercup, etc.). The exact class names are derived from the directory structure.\n",
        "\n",
        "Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rajatkumar30/flower-recognition) and place it in a local directory (e.g., `./data/train` and `./data/test`). Update the `train_path` and `test_path` variables accordingly."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Requirements\n",
        "\n",
        "- Python 3.8+\n",
        "- TensorFlow 2.x\n",
        "- Keras\n",
        "- NumPy\n",
        "- Matplotlib\n",
        "- Pillow (for image loading)\n",
        "\n",
        "Install dependencies via pip:\n",
        "\n",
        "```bash:disable-run
        "pip install tensorflow numpy matplotlib pillow\n",
        "```"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Import Libraries"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import os\n",
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "import matplotlib.pyplot as plt\n",
        "from tensorflow.keras.optimizers import Adam\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.applications import ResNet50\n",
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
        "from tensorflow.keras.applications.resnet50 import preprocess_input\n",
        "from tensorflow.keras.preprocessing.image import load_img, img_to_array\n",
        "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Set Paths and Hyperparameters"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "train_path = '/kaggle/input/17flowerclasses/17flowerclasses/train'\n",
        "test_path = '/kaggle/input/17flowerclasses/17flowerclasses/test'\n",
        "batch_size = 32\n",
        "epochs = 20"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Data Generators"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "train_datagen = ImageDataGenerator(\n",
        "    preprocessing_function=preprocess_input,\n",
        "    rotation_range=20,\n",
        "    width_shift_range=0.2,\n",
        "    height_shift_range=0.2,\n",
        "    shear_range=0.2,\n",
        "    zoom_range=0.2,\n",
        "    horizontal_flip=True,\n",
        "    validation_split=0.2\n",
        ")\n",
        "\n",
        "test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)\n",
        "\n",
        "train_generator = train_datagen.flow_from_directory(\n",
        "    train_path,\n",
        "    target_size=(224, 224),\n",
        "    batch_size=batch_size,\n",
        "    class_mode='categorical',\n",
        "    subset='training'\n",
        ")\n",
        "\n",
        "validation_generator = train_datagen.flow_from_directory(\n",
        "    train_path,\n",
        "    target_size=(224, 224),\n",
        "    batch_size=batch_size,\n",
        "    class_mode='categorical',\n",
        "    subset='validation'\n",
        ")\n",
        "\n",
        "test_generator = test_datagen.flow_from_directory(\n",
        "    test_path,\n",
        "    target_size=(224, 224),\n",
        "    batch_size=batch_size,\n",
        "    class_mode='categorical',\n",
        "    shuffle=False\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Build Model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))\n",
        "for layer in base_model.layers:\n",
        "    layer.trainable = False\n",
        "\n",
        "base_model.summary()\n",
        "\n",
        "model = Sequential([\n",
        "    base_model,\n",
        "    tf.keras.layers.GlobalAveragePooling2D(),\n",
        "    Dense(256, activation='relu'),\n",
        "    Dropout(0.5),\n",
        "    Dense(17, activation='softmax')\n",
        "])\n",
        "\n",
        "model.compile(optimizer=Adam(learning_rate=0.001),\n",
        "              loss='categorical_crossentropy',\n",
        "              metrics=['accuracy'])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Train Model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "history = model.fit(\n",
        "    train_generator,\n",
        "    epochs=epochs,\n",
        "    validation_data=validation_generator\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Plot Initial Training Accuracy"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "plt.plot(history.history['accuracy'], label='Train Accuracy')\n",
        "plt.plot(history.history['val_accuracy'], label='Validation Accuracy')\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.legend()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Fine-Tuning"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "for layer in base_model.layers[-10:]:\n",
        "    layer.trainable = True\n",
        "\n",
        "model.compile(\n",
        "    optimizer=Adam(learning_rate=1e-5),\n",
        "    loss='categorical_crossentropy',\n",
        "    metrics=['accuracy']\n",
        ")\n",
        "\n",
        "history_finetune = model.fit(\n",
        "    train_generator,\n",
        "    validation_data=validation_generator,\n",
        "    epochs=5\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Plot Fine-Tuning Accuracy"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "plt.plot(history_finetune.history['accuracy'], label='Train Accuracy')\n",
        "plt.plot(history_finetune.history['val_accuracy'], label='Validation Accuracy')\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.legend()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Prediction Function"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "labels = {v: k for k, v in train_generator.class_indices.items()}\n",
        "\n",
        "def predict_label(model, img_path):\n",
        "    img = load_img(img_path, target_size=(224, 224))\n",
        "    img_array = img_to_array(img)\n",
        "    img_array = np.expand_dims(img_array, axis=0)\n",
        "    img_array = preprocess_input(img_array)\n",
        "    pred = model.predict(img_array)\n",
        "    class_index = np.argmax(pred)\n",
        "    return labels[class_index]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Sample Prediction"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "img_path = \"/kaggle/input/flower/6028_SpanishBluebells_CGC8649sq-scaled.jpg\"\n",
        "result = predict_label(model, img_path)\n",
        "print(\"Predicted label:\", result)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Model Architecture\n",
        "\n",
        "- Base: ResNet50 (pre-trained on ImageNet, frozen initially).\n",
        "- Top Layers:\n",
        "  - Global Average Pooling.\n",
        "  - Dense (256 units, ReLU).\n",
        "  - Dropout (0.5).\n",
        "  - Dense (17 units, Softmax).\n",
        "- Optimizer: Adam.\n",
        "- Loss: Categorical Crossentropy.\n",
        "- Metrics: Accuracy.\n",
        "\n",
        "During fine-tuning, the last 10 layers of ResNet50 are unfrozen, and the learning rate is reduced to 1e-5."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Results\n",
        "\n",
        "- **Initial Training**: Expect training accuracy to improve over 20 epochs, with validation accuracy monitoring for overfitting.\n",
        "- **Fine-Tuning**: Further improvement in accuracy after unfreezing layers.\n",
        "- **Sample Prediction**: For the provided image (`6028_SpanishBluebells_CGC8649sq-scaled.jpg`), the model outputs the predicted class (e.g., \"Spanish Bluebells\").\n",
        "\n",
        "Example plots (generated during runtime):\n",
        "- Training vs. Validation Accuracy (initial and fine-tuned)."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Limitations\n",
        "\n",
        "- This notebook assumes a GPU-enabled environment (like Kaggle) for faster training. On CPU, reduce `batch_size` or epochs.\n",
        "- Dataset paths are Kaggle-specific; adapt for local use.\n",
        "- No model saving/exporting—add `model.save('flower_model.h5')` if needed."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Contributing\n",
        "\n",
        "Feel free to fork and submit pull requests for improvements, such as adding model evaluation on the test set or hyperparameter tuning."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## License\n",
        "\n",
        "This project is licensed under the MIT License."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Acknowledgments\n",
        "\n",
        "- Dataset: Provided by [rajatkumar30 on Kaggle](https://www.kaggle.com/datasets/rajatkumar30/flower-recognition).\n",
        "- Inspired by standard transfer learning practices for image classification."
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.10.13"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
```
