from abc import ABCMeta, abstractmethod

class ActivationFunc(metaclass=ABCMeta):

    @abstractmethod
    def function(self, Z):
        pass

    @abstractmethod
    def derivative(self, A, Z):
        pass