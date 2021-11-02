#--------------------Importations---------------------#
from base_layer import Layer
import numpy as np
#-----------------------------------------------------#

class Reshape(Layer):
    
    def __init__(self, input_shape, output_shape):
        
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, input):
        
        self.output = np.reshape(input, self.output_shape)
        
        return self.output
    
    def backward(self, grad_o, alpha):
        
        grad_i = np.reshape(grad_o, self.input_shape)
        
        return grad_i