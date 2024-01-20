#--------------------Importations---------------------#
from base_layer import Layer
import numpy as np
import time
#-----------------------------------------------------#

class Dense(Layer):
    def __init__(self, input_size, output_size):
        #----Matrice des poids----#
        self.w = 2*np.random.rand(output_size, input_size)-1
        #-------------------------#
        
        #----Matrice des biais----#
        self.b = 2*np.random.rand(output_size, 1)-1
        #-------------------------#
        
        #----Inertials terms-----#
        self.v_w = np.zeros((output_size, input_size))
        self.v_b = np.zeros((output_size, 1))
        #------------------------#
        
        #----Inertials terms-----#
        self.grad_sum_w = np.zeros((output_size, input_size))
        self.grad_sum_b = np.zeros((output_size, 1))
        #----Inertials terms-----#
        
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
    def backward(self, grad_o, alpha, grad_method):
                
        #----Gradient d'entrée----#
        grad_i = np.dot(self.w.T, grad_o)
        #-------------------------#
        
        #----Gradient des poids----#
        grad_w = np.dot(grad_o, self.input.T)
        #--------------------------#
        
        #----Momentum method----#
        gamma = 0
        if(grad_method == "momentum"):
            gamma = 0.9
        #----------------------#
        
        #----Adagrad method----#
        if(grad_method == "adagrad"):
            self.grad_sum_w += grad_w**2
            self.grad_sum_b += grad_o**2
            alpha_w = alpha / (1e-8 + np.sqrt(self.grad_sum_w))
            alpha_b = alpha / (1e-8 + np.sqrt(self.grad_sum_b))
            
            self.w = self.w - alpha_w * grad_w
            self.b = self.b - alpha_b * grad_o
            
            alpha = 0
        #----------------------#

        #----Update Inertial terms----#
        self.v_w = gamma * self.v_w + alpha * grad_w
        self.v_b = gamma * self.v_b + alpha * grad_o
        #-----------------------------#
        
        #----Application des gradients----#
        self.w = self.w - self.v_w
        self.b = self.b - self.v_b
        #---------------------------------#
    
        return grad_i