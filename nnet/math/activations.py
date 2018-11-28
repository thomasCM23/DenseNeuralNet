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
        less_than_zero = (Z <= 0).astype(np.int)
        greater_than_zero = (Z > 0).astype(np.int)
        dZ_temp = np.multiply(less_than_zero, 0)
        dZ = np.add(dZ_temp, greater_than_zero)
        # dZ = np.array(Z, copy=True)
        # dZ[Z <= 0] = 0
        return dZ


class LeakyRelu(ActivationFunc):

    def __init__(self, leaks=0.01):
        self.leaks = leaks

    def function(self, Z):
        return np.maximum(self.leaks * Z, Z)

    def derivative(self, A, Z, m):
        less_than_zero = (Z < 0).astype(np.int)
        greater_than_zero = (Z >= 0).astype(np.int)

        dZ_temp = np.multiply(less_than_zero,self.leaks)
        dZ = np.add(dZ_temp, greater_than_zero)
        return dZ


class Elu(ActivationFunc):

    def __init__(self, leaks=0.01):
        self.leaks = leaks

    def function(self, Z):
        less_than_zero = (Z < 0).astype(np.int)
        greater_than_zero = (Z >= 0).astype(np.int)

        Z_temp = np.multiply(less_than_zero, self.leaks * (np.exp(Z) - 1))
        Z = np.add(Z_temp, greater_than_zero * Z)
        return Z

    def derivative(self, A, Z, m):
        less_than_zero = (Z < 0).astype(np.int)
        greater_than_zero = (Z >= 0).astype(np.int)

        dZ_temp = np.multiply(less_than_zero, self.leaks * np.exp(Z))
        dZ = np.add(dZ_temp, greater_than_zero)
        return dZ


class Linear(ActivationFunc):

    def function(self, Z):
        return Z

    def derivative(self, A, Z, m):
        return A