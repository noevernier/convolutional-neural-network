from os import error
import sys
from colorama import Fore, Back, Style

sys.path.append("/Users/noe/Documents/School/MP*/Tipe/Informatique/CNN/NeuralNetwork")

from network import *

import numpy as np
import matplotlib.pyplot as plt

image_size = 28
image_pixels = image_size * image_size

data_path = "/Users/noe/Documents/School/MP*/Tipe/Informatique/CNN/Examples/MNIST/"
train_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
lr = np.arange(10)
train_labels = (lr==np.asfarray(train_data[:, :1])).astype(np.float)


x_test= np.reshape(train_imgs[0:1000], (1000, 1, image_size,image_size))
y_test = np.reshape(train_labels[0:1000], (1000, 10, 1))


network = Network.load_model(None, 'mnist')

y_test_predict = network.predict(x_test)
error = 0

deb = 0
fin = 800
for i in range(deb, fin):
    #pred = np.reshape(np.round(y_test_predict[i],2), (1,10))
    target = np.argmax(y_test[i])
    guess = np.argmax(y_test_predict[i])
    if(target != guess):
        print(Fore.RED + "[*] The Number : ", target, " - The Guess : ", guess)
        error +=1
    else:
        print(Fore.GREEN + "[*] The Number : ", target, " - The Guess : ", guess)
    #print('Prediction  : ', np.argmax(y_test_predict[i]), "Target : ", np.argmax(y_test[i]))

print(Fore.YELLOW, "[Toral Error] -> ", 100*error/(fin-deb), '%')

fig = plt.figure()
ax = fig.add_subplot(111)

#ax.scatter([i for i in range(len(network.errors))], network.errors, c=network.errors, cmap='twilight_shifted')
ax.plot([i for i in range(len(network.errors))], network.errors, color='red')
plt.savefig('/Users/noe/Documents/School/MP*/Tipe/Informatique/CNN/Examples/images/errors.png')