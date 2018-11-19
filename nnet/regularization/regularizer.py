from abc import ABCMeta, abstractmethod
import numpy as np

class Regularizer(metaclass=ABCMeta):

    def __init__(self, lamda=0):
        self.lamda = lamda

    @abstractmethod
    def regularizer_cost(self, layers, num_instances):
        return 0

    @abstractmethod
    def regularizer_backprop(self, layer_weight, num_instances):
        return 0


class L2Regularization(Regularizer):
    def __init__(self, lamda=0.1):
        super(L2Regularization, self).__init__(lamda)

    def regularizer_cost(self, layers, num_instances):
        reg_val = 0
        L = len(layers)

        for l in range(1, L):
            reg_val += np.sum(np.square(layers[l].W))

        return (self.lamda / (2 * num_instances)) * reg_val

    def regularizer_backprop(self, layer_weight, num_instances):
        return ((self.lamda/num_instances) * layer_weight)


class L1Regularization(Regularizer):
    def __init__(self, lamda=0.1):
        super(L1Regularization, self).__init__(lamda)

    def regularizer_cost(self, layers, num_instances):
        reg_val = 0
        L = len(layers)

        for l in range(1, L):
            reg_val += np.sum(layers[l].W)

        return (self.lamda / (2 * num_instances)) * reg_val

    def regularizer_backprop(self, layer_weight, num_instances):
        return self.lamda/ (2 *num_instances)