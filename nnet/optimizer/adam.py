from nnet.optimizer import OptimizationAlgo
import numpy as np

class Adam(OptimizationAlgo):

    def __init__(self, learning_rate = 0.01, beta_1 = .9, beta_2 = 0.999, epsilon = 1e-8):
        super(Adam, self).__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iteration_num = 0

    def update_parameters(self, layers):
        self.iteration_num += 1
        L = len(layers)

        for l in range(1, L):
            layers[l].vdW = self.beta_1 * layers[l].vdW + (1 - self.beta_1) * layers[l].dW
            layers[l].vdb = self.beta_1 * layers[l].vdb + (1 - self.beta_1) * layers[l].db

            layers[l].sdW = self.beta_2 * layers[l].sdW + (1 - self.beta_2) * (np.power(layers[l].dW, 2))
            layers[l].sdb = self.beta_2 * layers[l].sdb + (1 - self.beta_2) * (np.power(layers[l].db, 2))

            vdW_corrected = layers[l].vdW / (1 - self.beta_1 ** 2)
            vdb_corrected = layers[l].vdb / (1 - self.beta_1 ** 2)

            sdW_corrected = layers[l].sdW / (1 - self.beta_1 ** 2)
            sdb_corrected = layers[l].sdb / (1 - self.beta_1 ** 2)

            layers[l].W = layers[l].W - (self.learning_rate * (vdW_corrected / (np.sqrt(sdW_corrected) + self.epsilon)))
            layers[l].b = layers[l].b - (self.learning_rate * (vdb_corrected / (np.sqrt(sdb_corrected) + self.epsilon)))
