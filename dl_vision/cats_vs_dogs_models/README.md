# Cats v/s Dogs Models

Jupyter notebooks demonstrating training, fine-tuning, and interpretation of convolutional neural network models for binary classification of images of cats and dogs, using a subset of the Kaggle Cats and Dogs dataset. Includes a vanilla-CNN trained from scratch and fine-tuned SoTA models (VGG19 and ResNet50) pre-trained on ImageNet. 

## Status
[![Open Issues](https://img.shields.io/github/issues-raw/apallath/cats_vs_dogs_models)](https://github.com/apallath/cats_vs_dogs_models/issues)
[![Closed Issues](https://img.shields.io/github/issues-closed-raw/apallath/cats_vs_dogs_models)](https://github.com/apallath/cats_vs_dogs_models/issues)

## Models
- Vanilla CNN (baseline) - 71% acc [cats_vs_dots_cnn.ipynb](https://github.com/apallath/cats_vs_dogs_models/blob/main/cats_vs_dogs_CNN.ipynb)
- Vanilla CNN with data augmentation and dropout - 85% acc [cats_vs_dots_cnn.ipynb](https://github.com/apallath/cats_vs_dogs_models/blob/main/cats_vs_dogs_CNN.ipynb)
- Fine-tuned VGG19 model with data augmentation and dropout - 93% acc [cats_vs_dots_VGG19.ipynb](https://github.com/apallath/cats_vs_dogs_models/blob/main/cats_vs_dogs_VGG19.ipynb)
- Fine-tuned ResNet50 model with data augmentation and dropout - 96% acc [cats_vs_dots_ResNet50.ipynb](https://github.com/apallath/cats_vs_dogs_models/blob/main/cats_vs_dogs_ResNet50.ipynb)

## Model interpretation
- Visualizing layer activations on a test image for fine-tuned VGG19 and ResNet50 [cats_vs_dots_layer_activations.ipynb](https://github.com/apallath/cats_vs_dogs_models/blob/main/cats_vs_dogs_layer_activations.ipynb)
- Filter visualization using gradient-ascent for fine-tuned ResNet50 [cats_vs_dots_grad_ascent.ipynb](https://github.com/apallath/cats_vs_dogs_models/blob/main/cats_vs_dogs_grad_ascent.ipynb)
- Grad-CAM visualization for fine-tuned ResNet50 [cats_vs_dots_grad_cam.ipynb](https://github.com/apallath/cats_vs_dogs_models/blob/main/cats_vs_dogs_grad_cam.ipynb)

## References:
- Chollet's Deep Learning with Python 1e.
- Keras documentation.


