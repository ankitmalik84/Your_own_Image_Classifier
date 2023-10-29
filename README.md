# 🌼 Flower Image Classifier 🌸

This project aims to build an image classifier that can identify different species of flowers. The trained model can be used in various applications, such as a mobile app that can recognize and name flowers when provided with images. 

## 🚀 Getting Started

To get started with this project, you'll need to follow these steps:

1. **Load and preprocess the image dataset.**
2. **Train the image classifier using a pre-trained model.**
3. **Use the trained classifier to predict the content of images.**

## 🔧 Prerequisites

- **Python**
- **PyTorch**
- **torchvision**
- **Jupyter Notebook** (for running the project)
- **GPU** (recommended for faster training)

## 📊 Data Description

The dataset used for this project is split into three parts: training, validation, and testing. Each dataset requires specific preprocessing steps. The means and standard deviations of the images should be normalized to [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225], respectively.

## 🏗️ Building and Training the Classifier

To build and train the image classifier, follow these steps:

1. 🔄 **Load a pre-trained network**, such as VGG.
2. 📊 **Define a new, untrained feed-forward network as a classifier** with ReLU activations and dropout.
3. ⚙️ **Train the classifier layers using backpropagation** with the pre-trained network as a feature extractor.
4. 📈 **Monitor loss and accuracy on the validation set** to optimize hyperparameters.

## 🖼️ Class Prediction

You can use the trained model to make predictions for a given image. The `predict` function takes a path to an image and a model checkpoint and returns the top-k most probable classes along with their probabilities.

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
```


## 💾 Model Checkpoint
The trained model checkpoint, named checkpoint.pth, can be found [Here](https://drive.google.com/file/d/1GrnTD_ufY_s9iDkjdaPsRuhGT7apC7lA/view?usp=sharing). You can use this checkpoint to load the trained model for making predictions.

## 📝 Authors
**Ankit Malik**

## 🙏 Acknowledgments
This project is part of [certification Program](https://graduation.udacity.com/confirm/e/cc13cd48-0ae7-11ee-9d79-03ca8868a5bc).
Special thanks to the Udacity team for providing the project guidelines.
