#--------------------Importations---------------------#
from base_layer import Layer
import numpy as np
import skimage.measure
import math
from numba import jit, cuda
#-----------------------------------------------------#

class Pool(Layer):
    
    def __init__(self, factor):
        
        self.factor = factor
    
    def forward(self, input):
        self.input = input
        self.output = np.zeros((len(input), math.ceil(len(input[0])/self.factor), math.ceil(len(input[0])/self.factor)))
        for i in range(len(input)):
            self.output[i] = skimage.measure.block_reduce(input[i], (self.factor,self.factor), np.max)
        return self.output
    
    def apply_mask(self, x):
        mask = x == np.max(x)
        return mask
    
    def backward(self, grad_o, alpha, grad_method):
        
        grad_i = np.zeros(self.input.shape)
        depth, output_width, output_height = grad_o.shape
        for i in range(depth):
            for w in range(output_width):
                for h in range(output_height):
                    
                    x = h
                    x_dx = x + self.factor
                    y = w
                    y_dy = y + self.factor
                    
                    input_slice = self.input[i, x:x_dx, y:y_dy]
                    mask = self.apply_mask(input_slice)
                    grad_i[i, x:x_dx, y:y_dy] += np.multiply(mask, grad_o[i, h, w])         
        return grad_i
