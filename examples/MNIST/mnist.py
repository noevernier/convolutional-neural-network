import sys

sys.path.append("/Users/noe/Documents/School/MP*/Tipe/Informatique/CNN/NeuralNetwork")

from network import *

import numpy as np
import matplotlib.pyplot as plt
import math

#----Chargement de la base de données----#
image_size = 28
image_pixels = image_size * image_size

data_path = "/Users/noe/Documents/School/MP*/Tipe/Informatique/CNN/Examples/MNIST/"
train_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
lr = np.arange(10)
train_labels = (lr==np.asfarray(train_data[:, :1])).astype(np.float)


#----Donées----#
samples = 800 #Nombre de de donées pour l'entrainement
x_train = np.reshape(train_imgs[:samples], (samples, 1, image_size, image_size))
y_train = np.reshape(train_labels[:samples], (samples, 10, 1))
#--------------#

#------------------------------------------#


#----Création du modèle----#
"""layers = [
    Reshape((1, 28, 28), (1 * 28 * 28, 1)),
    Dense(1 * 28 * 28, 100),
    Activation(sigmoid, d_sigmoid),
    Dense(100, 50),
    Activation(sigmoid, d_sigmoid),
    Dense(50, 10),
    Activation(sigmoid, d_sigmoid)
]
"""
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