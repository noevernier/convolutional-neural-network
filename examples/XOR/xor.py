import sys

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

"""
def get_points():
    points = []
    for x in np.linspace(0, 1, 20):
        for y in np.linspace(0, 1, 20):
            z = network.predict(np.reshape([[x, y]], (1, 2, 1)))
            points.append([x, y, z[0][0][0]])

    points = np.array(points)
    return points


for e in range(100):
    network.train(x_train, y_train, epochs=1, learning_rate=0.1, get_info=False)
    points = get_points()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='twilight_shifted')
    plt.savefig('CNN/Examples/images/frame'+ str(e)+'.png')"""