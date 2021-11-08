# Deep learning models for text/natural language data [in `dl_nlp`]

WIP

# Deep learning models for image data [in `dl_vision/`]

## Transfer learning and interpretation on the Cats-vs-Dogs dataset (`cats_vs_dogs_models/`)

## Models
- Vanilla CNN (baseline) - 71% acc [cats_vs_dots_cnn.ipynb](https://github.com/apallath/cats_vs_dogs_models/blob/main/cats_vs_dogs_CNN.ipynb)
- Vanilla CNN with data augmentation and dropout - 85% acc [cats_vs_dots_cnn.ipynb](https://github.com/apallath/cats_vs_dogs_models/blob/main/cats_vs_dogs_CNN.ipynb)
- Fine-tuned VGG19 model with data augmentation and dropout - 93% acc [cats_vs_dots_VGG19.ipynb](https://github.com/apallath/cats_vs_dogs_models/blob/main/cats_vs_dogs_VGG19.ipynb)
- Fine-tuned ResNet50 model with data augmentation and dropout - 96% acc [cats_vs_dots_ResNet50.ipynb](https://github.com/apallath/cats_vs_dogs_models/blob/main/cats_vs_dogs_ResNet50.ipynb)

## Model interpretation
- Visualizing layer activations on a test image for fine-tuned VGG19 and ResNet50 [cats_vs_dots_layer_activations.ipynb](https://github.com/apallath/cats_vs_dogs_models/blob/main/cats_vs_dogs_layer_activations.ipynb)
- Filter visualization using gradient-ascent for fine-tuned ResNet50 [cats_vs_dots_grad_ascent.ipynb](https://github.com/apallath/cats_vs_dogs_models/blob/main/cats_vs_dogs_grad_ascent.ipynb)
- Grad-CAM visualization for fine-tuned ResNet50 [cats_vs_dots_grad_cam.ipynb](https://github.com/apallath/cats_vs_dogs_models/blob/main/cats_vs_dogs_grad_cam.ipynb)

## Deep Dream (`deep_dream/`)
- Deep Dream

## Classification and generation on the Fashion-MNIST dataset (`fashion_mnist_models/`)

### Classification models
- Vanilla CNN - 91% acc.

### Latent-space generative models
- Convolutional variational autoencoder for image generation

# Machine learning algorithms for tabular data [in `ml_tab/`]

## Classification (`supervised/classification_basic.ipynb`, `supervised/classification_adv.ipynb`)
- k-Nearest Neighbours classification
- Logistic Regression
- Logistic Regression using polynomial features
- SVM with linear kernel, binary classification, multi-class classification
- Decision Tree
- Gaussian Naive Bayes
- Random Forest

## Regression (`supervised/regression.ipynb`)
- k-Nearest Neighbours regression
- Linear Regression
- Ridge Regression (Regularization)
- Ridge Regression with feature scaling
- Polynomial Regression

## Clustering (`unsupervised/clustering.ipynb`)
- k-Means Clustering
- DBSCAN
- Spectral Clustering
- Markov Clustering (markov_clustering Python module)
