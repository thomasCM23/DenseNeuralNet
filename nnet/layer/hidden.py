from nnet.activation import Sigmoid
import numpy as np

class Hidden:

    def __init__(self, numUnitsPrevLayer, numUnits, initilization="he", activation=Sigmoid, name=None):

        if(initilization == "he"):
            initVal = np.sqrt(2/numUnitsPrevLayer)
        elif(initilization == "xavier"):
            initVal = np.sqrt(1/numUnitsPrevLayer)
        else:
            initVal = 0.1

        self.W = np.random.randn(numUnits, numUnitsPrevLayer) * initVal
        self.b = np.zeros(shape=(numUnits, 1))
        self.activation = activation
        self.name = name
        self.Z, self.A = None, None
        self.dW, self.db, self.dZ, self.dA = None, None, None, None

