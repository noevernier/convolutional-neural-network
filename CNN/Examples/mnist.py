import sys

sys.path.append("/Users/noe/Documents/School/MP*/Tipe/Informatique/CNN/NeuralNetwork")

from network import *

import numpy as np
import matplotlib.pyplot as plt

image_size = 28
image_pixels = image_size * image_size

data_path = "CNN/Examples/"
train_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
lr = np.arange(10)
train_labels = (lr==np.asfarray(train_data[:, :1])).astype(np.float)

    
x_train = np.reshape(train_imgs[:1000], (1000, image_pixels, 1))
y_train = np.reshape(train_labels[:1000], (1000, 10, 1))
img = x_train[0].reshape((28,28))

layers = [
    Dense(image_pixels, 40),
    Activation(tanh, d_tanh),
    Dense(40, 10),
    Activation(tanh, d_tanh)
]
network = Network(layers)
network.use(min_square, d_min_square)
network.train(x_train, y_train, epochs=1000, learning_rate=0.1, get_info=True)
