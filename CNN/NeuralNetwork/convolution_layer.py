#--------------------Importations---------------------#
from base_layer import Layer
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
#-----------------------------------------------------#

class Convolution(Layer):
    def __init__(self, input_shape, k_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.k_size = k_size
        
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - k_size + 1, input_width - k_size + 1)
        self.k_shape = (depth, input_depth, k_size, k_size)
        
        self.k = np.random.randn(*self.k_shape)
        self.b = np.random.randn(*self.output_shape)
    
    def forward(self, input):
        
        self.input = input
        self.output = np.copy(self.b)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.k[i, j], "valid")
            plt.show()
        return self.output
    
    def backward(self, grad_o, alpha):
        
        grad_k = np.zeros(self.k_shape)
        grad_i = np.zeros(self.input_shape)
        
        #----Gradient des kernels et entrée----#
        for i in range(self.depth):
            for j in range(self.input_depth):
                grad_k[i, j] = signal.correlate2d(self.input[j], grad_o[i], "valid")
                grad_i[j] += signal.convolve2d(grad_o[i], self.k[i, j], "full")
        #--------------------------#
        
        #----Application des gradients----#
        self.k = self.k - alpha * grad_k
        self.b = self.b - alpha * grad_o
        #---------------------------------#
        
        return grad_i