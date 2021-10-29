import numpy as np
#----Liste des fonctions de coût----#

#----Erreur quadratique moyenne---- (x-y)²#
def min_square(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def d_min_square(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;
#-----------------------------------------#