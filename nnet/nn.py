from nnet.activation import Sigmoid
from nnet.layer import Hidden
import numpy as np

class Net:

    def __init__(self, regularizationType="L2", lamdba = 0.1):
        self.regularization = regularizationType
        self.lamdba = lamdba
        self.layers = {}

    def hidden(self, perviousLayer, numOfUnits=1, initilization="he", activation=Sigmoid, name=None):
        if(name is None):
            name = len(self.layers)
        numUnitsPrevLayer = perviousLayer.shape[0]
        self.layers[name] = Hidden(numUnitsPrevLayer, numOfUnits, initilization, activation, name)
        return self.layers[name]

    def placeholder(self, shape=(1,1), name="X"):
        self.layers[name] = np.zeros(shape=shape)
        return self.layers["A0"]