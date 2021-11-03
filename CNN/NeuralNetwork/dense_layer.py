#--------------------Importations---------------------#
from base_layer import Layer
import numpy as np
import time
#-----------------------------------------------------#

class Dense(Layer):
    def __init__(self, input_size, output_size):
        #----Matrice des poids----#
        self.w = np.random.randn(output_size, input_size)
        #-------------------------#
        
        #----Matrice des biais----#
        self.b = np.random.randn(output_size, 1)
        #-------------------------#
        
    #---------------------------
    """Parcours d'une couche"""
    #---------------------------
    def forward(self, input):
        self.input = input
        #----Calcul : Output = W * Input + B ----#
        self.output = np.dot(self.w, self.input) + self.b
        #----------------------------------------#
        return self.output
    
    #---------------------------
    """Calcul des gradients"""
    #---------------------------
    def backward(self, grad_o, alpha):
                
        #----Gradient d'entrée----#
        grad_i = np.dot(self.w.T, grad_o)
        #-------------------------#
        
        #----Gradient des poids----#
        grad_w = np.dot(grad_o, self.input.T)
        #--------------------------#
        
        #----Application des gradients----#
        self.w = self.w - alpha * grad_w
        self.b = self.b - alpha * grad_o
        #---------------------------------#
    
        return grad_i