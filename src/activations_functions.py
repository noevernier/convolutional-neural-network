import numpy as np
#----Liste des fonctions d'activations----#

#----Tangente Hyperbolique---- tanh(x)#
def tanh(x):
    return np.tanh(x);

def d_tanh(x):
    return 1-np.tanh(x)**2;
#-------------------------------------#

#-----------------Sigmoid-------------#
def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return (1/(1+np.exp(-x))) * (1 - (1/(1+np.exp(-x))))
#-------------------------------------#

#-----------------RELU-------------#
def relu(x):
    return np.maximum(0,x)

def d_relu(x):
    return 1. * (x > 0)
#-------------------------------------#