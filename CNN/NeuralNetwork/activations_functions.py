import numpy as np
#----Liste des fonctions d'activations----#

#----Tangente Hyperbolique---- tanh(x)#
def tanh(x):
    return np.tanh(x);

def d_tanh(x):
    return 1-np.tanh(x)**2;