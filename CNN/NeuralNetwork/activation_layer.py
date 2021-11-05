
#--------------------Importations---------------------#
from base_layer import Layer
import numpy as np
#-----------------------------------------------------#

class Activation(Layer):
    def __init__(self, act_fun, d_act_fun):
        
        #----Fonction d'activation----#
        self.act_fun = act_fun
        self.d_act_fun = d_act_fun
        #-----------------------------#

    #---------------------------
    """Parcours d'une couche"""
    #---------------------------
    def forward(self, input):
        self.input = input
        #----Calcul : Output = f(Input) ----#
        self.output = self.act_fun(self.input)
        #----------------------------------------#
        return self.output
    
    #---------------------------
    """Calcul des gradients"""
    #---------------------------
    def backward(self, grad_o, alpha, grad_method):
        #----Gradient d'entrée----#
        grad_i = np.multiply(grad_o, self.d_act_fun(self.input))
        #-------------------------#
        return grad_i