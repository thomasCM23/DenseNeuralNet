from abc import ABCMeta, abstractmethod
import numpy as np

class ActivationFunc(metaclass=ABCMeta):

    @abstractmethod
    def function(self, Z):
        pass

    @abstractmethod
    def derivative(self, Z):
        pass


class Sigmoid(ActivationFunc):

    def function(self, Z):
        return 1 / ( 1 + np.exp(-Z))

    def derivative(self, Z):
        s = self.function(Z)
        return s * (1-s)


class Tanh(ActivationFunc):

    def function(self, Z):
        return np.tanh(Z)

    def derivative(self, Z):
        return 1 - np.power(self.function(Z), 2)


class Relu(ActivationFunc):

    def function(self, Z):
        return np.maximum(0, Z)

    def derivative(self, Z):
        Z[Z < 0 ] = 0
        Z[Z >= 0] = 1
        return Z


class LeakyRelu(ActivationFunc):

    def __init__(self, leaks=0.01):
        self.leaks = leaks

    def function(self, Z):
        return np.maximum(self.leaks * Z, Z)

    def derivative(self, Z):
        Z[Z < 0] = self.leaks
        Z[Z >= 0] = 1
        return Z


class Elu(ActivationFunc):

    def __init__(self, leaks=0.01):
        self.leaks = leaks

    def function(self, Z):
        Z[Z < 0] = self.leaks * (np.exp(Z) - 1)
        Z[Z >= 0] = Z
        return Z

    def derivative(self, Z):
        Z[Z < 0] = self.leaks * np.exp(Z)
        Z[Z >= 0] = 1
        return Z


class Linear(ActivationFunc):

    def function(self, Z):
        return Z

    def derivative(self, Z):
        return 1