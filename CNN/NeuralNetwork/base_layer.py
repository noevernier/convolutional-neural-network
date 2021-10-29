#----Structure fondamental d'une couche----#
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        #----Parcours----#
        pass

    def backward(self, grad_o, alpha):
        #----Retour avec calcul de gradient----#
        pass
