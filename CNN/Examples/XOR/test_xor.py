import sys
import json

sys.path.append("/Users/noe/Documents/School/MP*/Tipe/Informatique/CNN/NeuralNetwork")

from network import *

import numpy as np
import matplotlib.pyplot as plt

x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
y_train = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = Network.load_model(None, "xor")

def get_points():
    points = []
    for x in np.linspace(0, 1, 20):
        for y in np.linspace(0, 1, 20):
            z = network.predict(np.reshape([[x, y]], (1, 2, 1)))
            points.append([x, y, z[0][0][0]])

    points = np.array(points)
    return points

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
points = get_points()
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='twilight_shifted')
plt.show()
