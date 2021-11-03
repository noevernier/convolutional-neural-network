import sys

sys.path.append("/Users/noe/Documents/School/MP*/Tipe/Informatique/CNN/NeuralNetwork")

from network import *

import numpy as np
import matplotlib.pyplot as plt
import math

image_size = 28
image_pixels = image_size * image_size

data_path = "CNN/Examples/MNIST/"
train_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
lr = np.arange(10)
train_labels = (lr==np.asfarray(train_data[:, :1])).astype(np.float)

samples = 500
    
x_train = np.reshape(train_imgs[:samples], (samples, 1, image_size, image_size))
y_train = np.reshape(train_labels[:samples], (samples, 10, 1))

layers = [
    Convolution((1, 28, 28), 3, 5),
    Pool(4),
    Activation(relu, d_relu),
    Reshape((5, 7, 7), (5 * 7 * 7, 1)),
    Dense(5 * 7 * 7, 100),
    Activation(sigmoid, d_sigmoid),
    Dense(100, 10),
    Activation(sigmoid, d_sigmoid)
]
network = Network(layers)
network.use(log_loss, d_log_loss)
network.train(x_train, y_train, epochs=200, learning_rate=0.1, get_info=True)
network.save_model('mnist')
