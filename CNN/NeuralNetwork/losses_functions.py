import numpy as np
#----Liste des fonctions de coût----#

#----Erreur quadratique moyenne---- (x-y)²#
def min_square(x, y):
    return np.mean(np.power(x-y, 2))

def d_min_square(x, y):
    return 2*(y-x)/x.size
#-----------------------------------------#

#------------Cross Entropy----------------#
def log_loss(x, y):
    return np.mean(-x * np.log(y) - (1 - x) * np.log(1 - y))

def d_log_loss(x, y):
    return ((1 - x) / (1 - y) - x / y) / np.size(x)
#-----------------------------------------#