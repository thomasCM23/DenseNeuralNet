from abc import ABCMeta, abstractmethod

class OptimizationAlgo(metaclass=ABCMeta):

    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate

    @abstractmethod
    def update_parameters(self, layers):
        pass