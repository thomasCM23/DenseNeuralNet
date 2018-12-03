import nnet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, mean_squared_error



housing_data = fetch_california_housing()
X, y = housing_data.data,housing_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

X_train = X_train.T
X_test = X_test.T
y_train = np.reshape(y_train, (1, y_train.shape[0]))
y_test = np.reshape(y_test, (1, y_test.shape[0]))
print(X_train.shape)
print(X_train[:,0])
idx = np.random.randint(1000)
X_prediction_check = X_train[:, idx: idx+3]
y_prediction_check = y_train[:, idx: idx+3]

optimizer = nnet.GradientDescent(learning_rate=0.001)
regulizer = nnet.L2Regularization(lamda=0)
loss_func = nnet.MeanSquaredError()

newNet = nnet.Net(regularization=regulizer, optimizer=optimizer, cost_function=loss_func)

input = newNet.input_placeholder(shape=(8, None))
# Just showing all activations
hidden1 = newNet.dense(input, numOfUnits=10, activation=nnet.Relu(), initilization="he")
hidden2 = newNet.dense(hidden1, numOfUnits=8, activation=nnet.Relu(), initilization="he")
hidden3 = newNet.dense(hidden2, numOfUnits=5, activation=nnet.Relu(), initilization="mod_he")
hidden4 = newNet.dense(hidden3, numOfUnits=3, activation=nnet.Sigmoid(), initilization=None)
output = newNet.dense(hidden4, numOfUnits=1, activation=nnet.Linear(), initilization=None)

num_epochs = 500

costs = []

for epoch in range(num_epochs):
    _, loss = newNet.train(X_train, y_train)
    costs.append(loss)
    if( epoch % 10 == 0):
        predict_checking = newNet.predict(X_prediction_check)
        print("Predicted Result: ", predict_checking, " Predicted Shape: ", predict_checking.shape)
        print("Real Value: ", y_prediction_check)
        print("------ Epoch: ", epoch, " ------")
        print("MSE: ", loss)
        print("----------------------------")



pred_y = newNet.predict(X_test)
print("Shape Of Predictions: \t", pred_y.shape)
print("Mean Squared Log Error: \t", mean_squared_log_error(y_test, pred_y))
print("Mean Squared Error: \t", mean_squared_error(y_test, pred_y))


# make an agg figure
fig, ax = plt.subplots()
ax.plot(costs)
ax.set_title('Loss Over Epochs')
fig.canvas.draw()
# grab the pixel buffer and dump it into a numpy array
X = np.array(fig.canvas.renderer._renderer)

# now display the array X as an Axes in a new figure
plt.show()

