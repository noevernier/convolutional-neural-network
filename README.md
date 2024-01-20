# Convolutional Neural Network Library

This repository contains a small library for creating Convolutional Neural Networks (CNNs) and Neural Networks (NNs) in Python

## Features

- Create and train Convolutional Neural Networks
- Create and train Neural Networks
- Support for various activation functions
- Support for different types of layers (convolutional, pooling, fully connected)

## Usage

Here's an example of how to create and train a Convolutional Neural Network using this library:

```python
from network import *

import numpy as np
import matplotlib.pyplot as plt

x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
y_train = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

layers = [
    Dense(2, 3),
    Activation(tanh, d_tanh),
    Dense(3, 1),
    Activation(tanh, d_tanh)
]
network = Network(layers)
network.set_loss(min_square, d_min_square)
network.train(x_train, y_train, epochs=100, learning_rate=0.1, get_info=True, grad_method="")
network.save_model("xor_momentum")
```

```python
#----Donées----#
samples = 800 #Nombre de de donées pour l'entrainement
x_train = np.reshape(train_imgs[:samples], (samples, 1, image_size, image_size))
y_train = np.reshape(train_labels[:samples], (samples, 10, 1))
#--------------#

#------------------------------------------#


#----Création du modèle----#
layers = [
    Convolution((1, 28, 28), 3, 3),
    Activation(relu, d_relu),
    Reshape((3, 26, 26), (3 * 26 * 26, 1)),
    Dense(3 * 26 * 26, 50),
    Activation(sigmoid, d_sigmoid),
    Dense(50, 10),
    Activation(sigmoid, d_sigmoid)
]
network = Network(layers)
network.set_loss(log_loss, d_log_loss)
#---------------------------#

#----Entrainement, Enregistrement du modèle----#
network.train(x_train, y_train, epochs=100, learning_rate=0.1, get_info=True, grad_method="")
network.save_model('mnist')
#----------------------------------------------#
```
