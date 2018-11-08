from nnet.math import Sigmoid
from nnet.math import CrossEntropyLoss
from nnet.layer import Hidden
from nnet.layer import Input
from nnet.optimizer import GradientDescent
from nnet.regularization import L2Regularization
import numpy as np
import warnings


class Net:

    def __init__(self, regularizationType=L2Regularization(lamda=0), optimizer=GradientDescent(learning_rate=0.01),
                 cost_function=CrossEntropyLoss):
        self.regularization = regularizationType
        self.layers = {}
        self.optimizer = optimizer
        self.cost_function = cost_function

    def dense(self, perviousLayer, numOfUnits=1, initilization="he", activation=Sigmoid, keep_prob=1.0):
        name = len(self.layers)
        if(name == 0):
            warnings.warn("No input layer found in neural net!")
            name = 1
        numUnitsPrevLayer = perviousLayer.shape[0]
        self.layers[name] = Hidden(numUnitsPrevLayer, numOfUnits, initilization, activation, name, keep_prob)
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

            # If layer is dropout do the calculations
            if(np.isclose(self.layers[l].keep_prob, 1.0, atol=1e-08, equal_nan=False)):
                self.layers[l].D = np.random.rand(self.layers[l].A.shape[0], self.layers[l].A.shape[1])
                self.layers[l].D = self.layers[l].D < self.layers[l].keep_prob
                self.layers[l].A = self.layers[l].A * self.layers[l].D
                self.layers[l].A = self.layers[l].A / self.layers[l].keep_prob

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
            self.layers[l - 1].dA = np.dot(self.layers[l].W.T, self.layers[l].dZ)

            # If layer is dropout do the calculationss
            if (np.isclose(self.layers[l].keep_prob, 1.0, atol=1e-08, equal_nan=False)):
                self.layers[l - 1].dA = self.layers[l - 1].dA * self.layers[l - 1].D
                self.layers[l - 1].dA = self.layers[l - 1].dA / self.layers[l - 1].keep_prob

