from nnet.math import Sigmoid
from nnet.math import CrossEntropyLoss
from nnet.layer import Hidden
from nnet.layer import Input
from nnet.optimizer import GradientDescent
from nnet.regularization import L2Regularization
import numpy as np
import warnings


class Net:

    def __init__(self, regularization=L2Regularization(lamda=0), optimizer=GradientDescent(learning_rate=0.01),
                 cost_function=CrossEntropyLoss()):
        self.regularization = regularization
        self.layers = {}
        self.optimizer = optimizer
        self.cost_function = cost_function
        self.L = 0
        self.m = 0

    def dense(self, perviousLayer, numOfUnits=1, initilization="he", activation=Sigmoid(), keep_prob=1.0):
        name = len(self.layers)
        if(name == 0):
            warnings.warn("No input layer found in neural net!")
            name = 1
        numUnitsPrevLayer = perviousLayer.shape[0]
        self.layers[name] = Hidden(numUnitsPrevLayer, numOfUnits, initilization, activation, name, keep_prob)
        return self.layers[name]

    def input_placeholder(self, shape=(1, None)):
        self.layers[0] = Input(shape=shape, name=0)
        return self.layers[0]

    def _forward_prop(self):
        for l in range(1, self.L):
            A_prev = self.layers[l-1].A
            self.layers[l].Z = np.dot(self.layers[l].W, A_prev) + self.layers[l].b
            self.layers[l].A = self.layers[l].activation.function(self.layers[l].Z)

            # If layer is dropout do the calculations
            if(self.layers[l].is_drop_out):
                self.layers[l].D = np.random.rand(self.layers[l].A.shape[0], self.layers[l].A.shape[1])
                self.layers[l].D = self.layers[l].D < self.layers[l].keep_prob
                self.layers[l].A = self.layers[l].A * self.layers[l].D
                self.layers[l].A = self.layers[l].A / self.layers[l].keep_prob

    def _backward_prop(self, AL, Y):
        # cost derivative
        self.layers[self.L-1].dA = - self.cost_function.derivative(AL, Y, 0)
        # doing back prop fro each layer
        for l in reversed(range(self.L)):
            if(l == 0): break
            self.layers[l].dZ = np.multiply(self.layers[l].dA,
                                            self.layers[l].activation.derivative(self.layers[l].A, self.layers[l].Z, 0))
            self.layers[l].dW = (1/self.m) * np.dot(self.layers[l].dZ, self.layers[l-1].A.T)
            self.layers[l].db = (1/self.m) * np.sum(self.layers[l].dZ, axis=1, keepdims=True)
            if (l == 0): break
            self.layers[l - 1].dA = np.dot(self.layers[l].W.T, self.layers[l].dZ)

            # If layer is dropout do the calculationss
            if (self.layers[l].is_drop_out):
                self.layers[l - 1].dA = self.layers[l - 1].dA * self.layers[l - 1].D
                self.layers[l - 1].dA = self.layers[l - 1].dA / self.layers[l - 1].keep_prob

    def _compute_cost(self, AL, Y):
        cost = self.cost_function.function(Z={"Y":Y, "AL":AL, "m":self.m}) + self.regularization.regularizer_cost(
            layers=self.layers, num_instances=self.m)
        cost = np.squeeze(cost)
        return cost

    def train(self, X, Y):

        self.layers[0].A = X
        self.m = X.shape[1]
        self.L = len(self.layers)
        self._forward_prop()
        cost = self._compute_cost(self.layers[self.L - 1].A, Y)
        self._backward_prop(self.layers[self.L - 1].A, Y)
        return self.layers, cost