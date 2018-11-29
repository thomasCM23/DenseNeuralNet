from nnet.math.functions import ActivationFunc
import numpy as np


class CrossEntropyLoss(ActivationFunc):

    def __init__(self):
        pass

    def function(self, Z):
        return (-np.dot(Z["Y"], np.log(Z["AL"]).T) - np.dot(1 - Z["Y"], np.log(1 - Z["AL"]).T)) / Z['m']

    def derivative(self, AL, Y, m):
        return - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))


class SoftmaxCrossEntropyLoss(ActivationFunc):

    def __init__(self):
        pass

    def function(self, Z):
        logprobs = np.multiply(Z["Y"], np.log(Z["AL"]))
        return (-np.sum(logprobs))/ Z['m']

    def derivative(self, AL, Y, m):
        return (AL - Y)


class MeanSquaredError(ActivationFunc):

    def __init__(self):
        pass

    def function(self, Z):
        squared_variations = (Z["AL"] - Z["Y"]) ** 2
        return  np.sum(squared_variations) / (2 * Z["m"])

    def derivative(self, AL, Y, m):
        variation = AL - Y
        return (2 * np.sum(variation)) / m
