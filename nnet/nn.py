from nnet.math import Sigmoid
from nnet.math import CrossEntropyLoss
from nnet.layer import Hidden
from nnet.layer import Input
import numpy as np
import warnings

class Net:

    def __init__(self, regularizationType="L2", lamdba=0.1, optimizer="adam", cost_function=CrossEntropyLoss):
        self.regularization = regularizationType
        self.lamdba = lamdba
        self.layers = {}
        self.optimizer = optimizer
        self.cost_function = cost_function

    def dense(self, perviousLayer, numOfUnits=1, initilization="he", activation=Sigmoid):
        name = len(self.layers)
        if(name == 0):
            warnings.warn("No input layer found in neural net!")
            name = 1
        numUnitsPrevLayer = perviousLayer.shape[0]
        self.layers[name] = Hidden(numUnitsPrevLayer, numOfUnits, initilization, activation, name)
        return self.layers[name]

    def input_placeholder(self, shape=(1, None), name="0"):
        self.layers[name] = Input(shape=shape, name=name)
        return self.layers[name]

    def _forward_prop(self):
        L = len(self.layers)
        for l in range(1, L):
            A_prev = self.layers[l-1].A
            self.layers[l].Z = np.dot(self.layers[l].W, A_prev) + self.layers[l].b
            self.layers[l].A = self.layers[l].activation.function(self.layers[l].Z)

    def _compute_cost(self):
        pass

    def _backward_prop(self):

        # cost derivative

        #####
        L = len(self.layers)
        m = self.layers["A0"].shape[1]
        for l in reversed(range(L-1)):
            self.layers[l].dZ = np.multiply(self.layers[l].A, self.layers[l].activation.derivative(self.layers[l].Z))
            self.layers[l].dW = (1/m) * np.dot(self.layers[l].dZ, self.layers[l-1].A.T)
            self.layers[l].db = (1/m) * np.sum(self.layers[l].dZ, axis=1, keepdims=True)
            self.layers[l - 1].A = np.dot(self.layers[l].W.T, self.layers[l].dZ)
