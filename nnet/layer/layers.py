from nnet.math import Sigmoid
from nnet.math import Linear
import numpy as np
from abc import ABCMeta, abstractmethod

class Layer(metaclass=ABCMeta):
    pass


class Hidden(Layer):

    def __init__(self, numUnitsPrevLayer, numUnits, initilization="he", activation=Sigmoid(), name=None, keep_prob=1.0):

        if(initilization == "he"):
            initVal = np.sqrt(2/numUnitsPrevLayer)
        elif(initilization == "xavier"):
            initVal = np.sqrt(1/numUnitsPrevLayer)
        else:
            initVal = 0.1 # reduce standard normal

        self.shape = (numUnits, numUnitsPrevLayer)
        self.keep_prob = keep_prob
        self.is_drop_out = self.keep_prob < 1.0
        self.W = np.random.randn(numUnits, numUnitsPrevLayer) * initVal
        self.b = np.zeros(shape=(numUnits, 1))
        self.activation = activation
        self.name = name
        self.Z, self.A = None, None
        self.D = None
        self.dW, self.db, self.dZ, self.dA = None, None, None, None
        self.vdW, self.vdb = np.zeros(shape=self.W.shape), np.zeros(shape=self.b.shape)
        self.sdW, self.sdb = np.zeros(shape=self.W.shape), np.zeros(shape=self.b.shape)


class Input(Layer):

    def __init__(self, shape=(None, 1), name=None):
        self.shape = shape
        self.name = name
        self.Z, self.A = None, None
        self.dW, self.db, self.dZ, self.dA = None, None, None, None
        self.is_drop_out = False
        self.activation = Linear()

    def set_A(self, X):
        self.A = X