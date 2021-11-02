from os import error
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


x_test= np.reshape(train_imgs[0:10000], (10000, 1, image_size,image_size))
y_test = np.reshape(train_labels[0:10000], (10000, 10, 1))


network = Network.load_model(None, 'mnist')

y_test_predict = network.predict(x_test)
error = 0
for i in range(3000, 3100):
    pred = np.reshape(np.round(y_test_predict[i],2), (1,10))
    print(pred)
    if(np.argmax(y_test_predict[i]) != np.argmax(y_test[i])):
        error +=1
    #print('Prediction  : ', np.argmax(y_test_predict[i]), "Target : ", np.argmax(y_test[i]))

print("[Toral Error] -> ", error, '%')

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter([i for i in range(len(network.errors))], network.errors, c=network.errors, cmap='twilight_shifted')
plt.savefig('CNN/Examples/images/errors.png')