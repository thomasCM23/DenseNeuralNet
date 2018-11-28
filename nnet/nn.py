from nnet.math import Sigmoid
from nnet.math import CrossEntropyLoss
from nnet.layer import Hidden
from nnet.layer import Input
from nnet.optimizer import GradientDescent
from nnet.regularization import L2Regularization
import numpy as np
import warnings


class Net:

    def __init__(self, regularization=L2Regularization(lamda=0), optimizer=GradientDescent(learning_rate=0.01),
                 cost_function=CrossEntropyLoss()):
        self.regularization = regularization
        self.layers = {}
        self.optimizer = optimizer
        self.cost_function = cost_function
        self.L = 0
        self.m = 0

    def dense(self, perviousLayer, numOfUnits=1, initilization="he", activation=Sigmoid(), keep_prob=1.0):
        name = len(self.layers)
        if(name == 0):
            warnings.warn("No input layer found in neural net!")
            name = 1
        numUnitsPrevLayer = perviousLayer.shape[0]
        self.layers[name] = Hidden(numUnitsPrevLayer, numOfUnits, initilization, activation, name, keep_prob)
        return self.layers[name]

    def input_placeholder(self, shape=(1, None)):
        self.layers[0] = Input(shape=shape, name=0)
        return self.layers[0]

    def _forward_prop(self, is_prediction = False):
        for l in range(1, self.L):
            A_prev = self.layers[l-1].A
            self.layers[l].Z = np.dot(self.layers[l].W, A_prev) + self.layers[l].b
            self.layers[l].A = self.layers[l].activation.function(self.layers[l].Z)

            # If layer is dropout do the calculations
            if(self.layers[l].is_drop_out and (not is_prediction)):
                self.layers[l].D = np.random.rand(self.layers[l].A.shape[0], self.layers[l].A.shape[1])
                self.layers[l].D = self.layers[l].D < self.layers[l].keep_prob
                self.layers[l].A = self.layers[l].A * self.layers[l].D
                self.layers[l].A = self.layers[l].A / self.layers[l].keep_prob


    def _backward_prop(self, AL, Y):
        # cost derivative
        self.layers[self.L-1].dA = self.cost_function.derivative(AL, Y, self.m)
        # doing back prop fro each layer
        for l in reversed(range(self.L)):
            if(l == 0): break

            self.layers[l].dZ = np.multiply(self.layers[l].dA,
                                            self.layers[l].activation.derivative(1, self.layers[l].Z, 0))
            self.layers[l].dW = (1/self.m) * np.dot(self.layers[l].dZ, self.layers[l-1].A.T) + \
                                self.regularization.regularizer_cost(layers=self.layers, num_instances=self.m)
            self.layers[l].db = (1/self.m) * np.sum(self.layers[l].dZ, axis=1, keepdims=True)
            if (l == 0): break
            self.layers[l - 1].dA = np.dot(self.layers[l].W.T, self.layers[l].dZ)

            # If layer is dropout do the calculationss
            if (self.layers[l].is_drop_out):
                self.layers[l - 1].dA = self.layers[l - 1].dA * self.layers[l - 1].D
                self.layers[l - 1].dA = self.layers[l - 1].dA / self.layers[l - 1].keep_prob

    def _compute_cost(self, AL, Y):
        cost = self.cost_function.function(Z={"Y":Y, "AL":AL, "m":self.m})
        cost = np.squeeze(cost)
        return cost

    def train(self, X, Y, _check_gradients = False):
        self.layers[0].A = X
        self.m = X.shape[1]
        self.L = len(self.layers)
        self._forward_prop()
        cost = self._compute_cost(self.layers[self.L - 1].A, Y)
        self._backward_prop(self.layers[self.L - 1].A, Y)

        if(_check_gradients):
            self._gradient_checking()

        self.layers = self.optimizer.update_parameters(self.layers)
        return self.layers, cost

    def predict(self, X):
        self.layers[0].A = X
        self._forward_prop(is_prediction=True)
        return (self.layers[self.L - 1].A)

    def _gradient_checking(self, epsilon = 1e-7):
        for l in range(1, self.L):
            theta = self.layers[l].W + self.layers[l].b
            # Approximate the gradients
            J_plus = self.layers[l].activation.function(theta + epsilon)
            J_minus = self.layers[l].activation.function(theta - epsilon)
            gradapprox = (J_plus - J_minus) / (2 * epsilon)
            # Calculate gradient
            gradient = self.layers[l].activation.derivative(1, theta, 0)
            # Calculate difference
            numerator = np.linalg.norm(gradient - gradapprox)  # Step 1'
            denominator = np.linalg.norm(gradient) + np.linalg.norm(gradapprox)  # Step 2'
            difference = numerator / denominator

            if difference < 1e-7:
                print("****************** The gradient is correct! ******************",
                      self.layers[l].activation.__class__.__name__
                      )
            else:
                print("xxxxxxxxxxxxxxxxxx The gradient is wrong! xxxxxxxxxxxxxxxxxx",
                      self.layers[l].activation.__class__.__name__
                      )
                print("Layer: ", l)
                print("Gradient Approximated: ", gradapprox)
                print("Calculated Grad: ", gradient)

        # Checking the gradient of cost function
        thetaL = self.layers[self.L - 1].A
        J_plus = self.cost_function.function(Z={"Y": 1, "AL": thetaL + epsilon, "m": self.m})
        J_minus = self.cost_function.function(Z={"Y": 1, "AL": thetaL - epsilon, "m": self.m})
        gradapprox = (J_plus - J_minus) / (2 * epsilon)
        gradient = np.sum(self.cost_function.derivative(thetaL, 1, self.m)) / self.m
        numerator = np.linalg.norm(gradient - gradapprox)  # Step 1'
        denominator = np.linalg.norm(gradient) + np.linalg.norm(gradapprox)  # Step 2'
        difference = numerator / denominator
        if difference < 1e-7:
            print("****************** The gradient is correct! Cost function ******************",
                   self.cost_function.__class__.__name__
                   )
        else:
            print("xxxxxxxxxxxxxxxxxx The gradient is wrong! Cost function xxxxxxxxxxxxxxxxxx",
                  self.cost_function.__class__.__name__
                  )
            print("Gradient Approximated: ", gradapprox)
            print("Calculated Grad: ", gradient)