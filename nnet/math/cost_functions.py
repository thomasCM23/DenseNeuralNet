from nnet.math.functions import ActivationFunc
import numpy as np


class CrossEntropyLoss(ActivationFunc):

    def __init__(self):
        pass

    def function(self, Z):
        logprobs = np.multiply(np.log(Z["AL"]), Z["Y"]) + np.multiply((1 - Z["Y"]), np.log(1 - Z["AL"]))
        return -(np.sum(logprobs) / Z["m"])

    def derivative(self, AL, Y):
        return np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)


class SoftmaxCrossEntropyLoss(ActivationFunc):

    def __init__(self):
        pass

    def function(self, Z):
        logprobs = np.multiply(Z["Y"], np.log(Z["AL"]))
        return -(np.sum(logprobs))

    def derivative(self, AL, Y):
        return AL - Y