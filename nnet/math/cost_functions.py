from nnet.math.functions import ActivationFunc
import numpy as np

class CrossEntropyLoss(ActivationFunc):

    def function(self, Z):
        return 1 / ( 1 + np.exp(-Z))

    def derivative(self, Z):
        s = self.function(Z)
        return s * (1-s)