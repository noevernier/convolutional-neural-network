# -----------------------------------------------------------
# NeuralNetwork Library
#
# (C) 2021 Noé Vernier, France
# -----------------------------------------------------------

#--------------------Importations---------------------#
from dense_layer import Dense
from activation_layer import Activation
from convolution_layer import Convolution
from reshape_layer import Reshape
from pool_layer import Pool

from activations_functions import tanh, d_tanh, sigmoid, d_sigmoid, relu, d_relu
from losses_functions import min_square, d_min_square, log_loss, d_log_loss

import pickle
import numpy as np
import time
#-----------------------------------------------------#

#----Classe Réseaux de Neurones----#
class Network:
    def __init__(self, layers):
        self.layers = layers
        self.loss = None
        self.d_loss = None
        self.errors = []
    #-------------------------------
    """Intialise la fontions coût"""
    #-------------------------------
    def set_loss(self, loss, d_loss):
        self.loss = loss
        self.d_loss = d_loss

    #-----------------------------------
    """Parcours du réseau de neurones"""
    #-----------------------------------
    def predict(self, inputs):
        n_inputs = len(inputs)
        outputs = []

        for i in range(n_inputs): #<-------------Pour chaque jeu de données
            output = inputs[i]
            for layer in self.layers: #<---------On parcours Iterativement chaque couche
                output = layer.forward(output)
            outputs.append(output)

        return outputs
    #---------------------------------------
    """Entrainement du réseau de neurones"""
    #---------------------------------------
    def train(self, inputs, targets, epochs, learning_rate, get_info, grad_method):
        #----Nombre de jeu de données----#
        n_inputs = len(inputs)
        #--------------------------------#
        #----Nombre d'iteration----#
        for e in range(epochs):
            error = 0
            #----Pour chaque jeu de données----#
            for j in range(n_inputs):
                
                #----Calcul de la sortie à comparer----#
                output = inputs[j]
                for layer in self.layers:
                    output = layer.forward(output)
                #--------------------------------------#
                
                #----Erreur commise pour ce jeu de données----#
                #if(np.argmax(targets[j]) != np.argmax(output)):
                #    error += 1
                error += self.loss(targets[j], output)
                #---------------------------------------------#
                
                #----Calcul du gradient pour chaques couches----#
                grad = self.d_loss(targets[j], output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate, grad_method) #<--- Application du gradient
                #-----------------------------------------------#
            #-----------------------------------#
            
            #----Affichage simple----#
            error /= n_inputs
            self.errors.append(error)
            if(get_info and e%1 == 0):
                print('[Traning] -> ', round(100*e/epochs,1), '%', '| [Error] ->', round(error,4))
            #------------------------#
        #----Fin Nombre d'iteration----#

    #-------------------------
    """Enregistre le model"""
    #-------------------------
    def save_model(self, model_name):
        pickle.dump(self, file = open(model_name+'.pickle', "wb"))

    #----------------------
    """Charge un model"""
    #----------------------
    def load_model(self, model):
        return pickle.load(open(model+'.pickle', "rb"))
