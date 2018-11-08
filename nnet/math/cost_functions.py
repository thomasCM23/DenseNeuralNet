from nnet.math.functions import ActivationFunc
import numpy as np

class CrossEntropyLoss(ActivationFunc):

    def __init__(self):
        pass

    def function(self, Z):
        logprobs = np.multiply(np.log(Z["AL"]), Z["Y"]) + np.multiply((1 - Z["Y"]), np.log(1 - Z["AL"]))
        return -(np.sum(logprobs) / Z["m"])

    def derivative(self, dA, Z):
        return np.divide(Z, dA) - np.divide(1 - Z, 1 - dA)