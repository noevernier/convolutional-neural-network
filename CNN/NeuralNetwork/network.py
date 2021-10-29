# -----------------------------------------------------------
# NeuralNetwork Library
#
# (C) 2021 Noé Vernier, France
# -----------------------------------------------------------

#--------------------Importations---------------------#
from dense_layer import Dense
from activation_layer import Activation

from activations_functions import tanh, d_tanh
from losses_functions import min_square, d_min_square
#-----------------------------------------------------#

#----Classe Réseaux de Neurones----#
class Network:
    def __init__(self, layers):
        self.layers = layers
        self.loss = None
        self.d_loss = None
    #-------------------------------
    """Intialise la fontions coût"""
    #-------------------------------
    def use(self, loss, d_loss):
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
    def train(self, inputs, targets, epochs, learning_rate, get_info):
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
                error += self.loss(targets[j], output)
                #---------------------------------------------#
                
                #----Calcul du gradient pour chaques couches----#
                grad = self.d_loss(targets[j], output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate) #<--- Application du gradient
                #-----------------------------------------------#
            #-----------------------------------#
            
            #----Affichage simple----#
            error /= n_inputs
            if(get_info and e%100 == 0):
                print('[Traning] -> ', 100*e/epochs, '%', '| [Error] ->', round(error,4))
            #------------------------#
        #----Fin Nombre d'iteration----#
