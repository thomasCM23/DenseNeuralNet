from nnet.optimizer import OptimizationAlgo

class Momentum(OptimizationAlgo):

    def __init__(self, learning_rate = 0.01, beta = .9):
        super(Momentum, self).__init__(learning_rate)
        self.beta = beta

    def update_parameters(self, layers):
        L = len(layers)

        for l in range(1, L):
            layers[l].vdW = self.beta * layers[l].vdW + (1 - self.beta) * layers[l].dW
            layers[l].vdb = self.beta * layers[l].vdb + (1 - self.beta) * layers[l].db
            layers[l].W = layers[l].W - (self.learning_rate * layers[l].vdW)
            layers[l].b = layers[l].b - (self.learning_rate * layers[l].vdb)
        return layers