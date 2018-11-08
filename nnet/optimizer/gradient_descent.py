from nnet.optimizer import OptimizationAlgo

class GradientDescent(OptimizationAlgo):

    def __init__(self, learning_rate = 0.01):
        super(GradientDescent, self).__init__(learning_rate)

    def update_parameters(self, layers):
        L = len(layers)

        for l in range(1, L):
            layers[l].W = layers[l].W - (self.learning_rate * layers[l].dW)
            layers[l].b = layers[l].b - (self.learning_rate * layers[l].db)
