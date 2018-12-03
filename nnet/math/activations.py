from nnet.math.functions import ActivationFunc
import numpy as np

class Sigmoid(ActivationFunc):

    def function(self, Z):
        return 1 / ( 1 + np.exp(-Z))

    def derivative(self, A, Z, m):
        s = self.function(Z)
        return s * (1 - s)


class Tanh(ActivationFunc):

    def function(self, Z):
        return np.tanh(Z)

    def derivative(self, A, Z, m):
        return (1 - np.power(self.function(Z), 2))


class Relu(ActivationFunc):

    def function(self, Z):
        return np.maximum(0, Z)

    def derivative(self, A, Z, m):
        dZ = np.copy(Z)
        dZ[Z <= 0] = 0
        dZ[Z > 0] = 1
        return dZ


class LeakyRelu(ActivationFunc):

    def __init__(self, leaks=0.1):
        self.leaks = leaks

    def function(self, Z):
        return np.maximum(self.leaks * Z, Z)

    def derivative(self, A, Z, m):
        dZ = np.copy(Z)
        dZ[Z <= 0] = self.leaks
        dZ[Z > 0] = 1
        return dZ


class Elu(ActivationFunc):

    def __init__(self, leaks=0.01):
        self.leaks = leaks

    def function(self, Z):
        A = np.copy(Z)
        A[Z <= 0] = self.leaks * (np.exp(Z) - 1)
        A[Z > 0] = Z
        return A

    def derivative(self, A, Z, m):
        dZ = np.copy(Z)
        dZ[Z <= 0] = self.leaks * np.exp(Z)
        dZ[Z > 0] = 1
        return dZ


class Linear(ActivationFunc):

    def function(self, Z):
        return Z

    def derivative(self, A, Z, m):
        return np.ones(Z.shape)


class Softmax(ActivationFunc):

    def function(self, Z):
        t = np.exp(Z)
        return t / np.sum(t, axis=0)

    def derivative(self, A, Z, m):
        return 1