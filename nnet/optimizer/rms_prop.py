from nnet.optimizer import OptimizationAlgo
import numpy as np

class RMSProp(OptimizationAlgo):

    def __init__(self, learning_rate = 0.01, beta = .9, epsilon = 1e-8):
        super(RMSProp, self).__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon

    def update_parameters(self, layers):
        L = len(layers)

        for l in range(1, L):
            layers[l].sdW = self.beta * layers[l].sdW + (1 - self.beta) * (np.power(layers[l].dW, 2))
            layers[l].sdb = self.beta * layers[l].sdb + (1 - self.beta) * (np.power(layers[l].db,2))
            layers[l].W = layers[l].W - (self.learning_rate * (layers[l].dW / (np.sqrt(layers[l].sdW) + self.epsilon)))
            layers[l].b = layers[l].b - (self.learning_rate * (layers[l].db / (np.sqrt(layers[l].sdb) + self.epsilon)))
