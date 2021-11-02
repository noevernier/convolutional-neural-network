from os import error
import sys

sys.path.append("/Users/noe/Documents/School/MP*/Tipe/Informatique/CNN/NeuralNetwork")

from network import *

import numpy as np
import matplotlib.pyplot as plt
import json

with open('CNN/Examples/MNIST/output.json') as f:
  data = json.loads(json.load(f))
  
user = np.resize(np.array(data), (1,1, 28, 28))


network = Network.load_model(None, 'mnist')
prediction = network.predict(user)

output = np.argmax(np.reshape(np.round(prediction[0],2), (1,10)))

print(output)