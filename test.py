import nnet
from nnet.optimizer import Momentum, Adam, RMSProp, GradientDescent
from nnet.regularization import L2Regularization
from nnet.math.cost_functions import MeanSquaredError, CrossEntropyLoss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def plot_dataset(X, y, axes):
    plt.figure(0)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

def plot_predition(X, y, y_pred, axes):
    plt.figure(1)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.plot(X[:, 0][y_pred == 0], X[:, 1][y_pred == 0], "cx")
    plt.plot(X[:, 0][y_pred == 1], X[:, 1][y_pred == 1], "yv")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)



X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# scatter plot, dots colored by class value
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])



optimizer = Adam(learning_rate=0.001)
regulizer = L2Regularization(lamda=0.5)
loss_func = CrossEntropyLoss()

newNet = nnet.Net(regularization=regulizer, optimizer=optimizer, cost_function=loss_func)

input = newNet.input_placeholder(shape=(2, None))
# Just showing all activations
hidden1 = newNet.dense(input, numOfUnits=15, activation=nnet.Elu())
hidden2 = newNet.dense(hidden1, numOfUnits=10, activation=nnet.LeakyRelu())
hidden3 = newNet.dense(hidden2, numOfUnits=8, activation=nnet.Relu())
hidden4 = newNet.dense(hidden3, numOfUnits=5, activation=nnet.Tanh())
output = newNet.dense(hidden4, numOfUnits=1, activation=nnet.Sigmoid())

num_epochs = 500

costs = []

for epoch in range(1, num_epochs):
    _, loss = newNet.train(np.array(X_train).T, np.array(y_train).T)
    costs.append(loss)
    if( epoch % 50 == 0):
        print("------ Epoch: ", epoch, " ------")
        print("Corss entropy Loss: ", loss)
        print("----------------------------")


pred_y = newNet.predict(np.array(X_test).T)
pred_y = np.reshape(pred_y, (pred_y.shape[0],))

# If Prediction is over .5 then class 1
pred_y[pred_y >= .5] = 1
pred_y[pred_y < .5] = 0

print("Incorrect Predictions: ", X_test[:, :][y_test != pred_y])

plot_predition(X_test, y_test, pred_y, [-1.5, 2.5, -1, 1.5])

print("F1 score: ", f1_score(y_test, pred_y))


# make an agg figure
fig, ax = plt.subplots()
ax.plot(costs)
ax.set_title('Loss Over Epochs')
fig.canvas.draw()
# grab the pixel buffer and dump it into a numpy array
X = np.array(fig.canvas.renderer._renderer)

# now display the array X as an Axes in a new figure
plt.show()