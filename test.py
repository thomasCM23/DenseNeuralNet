import nnet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score



digits = fetch_mldata("MNIST original")

X, y = digits["data"],digits["target"]
y = y.astype(np.int)
print(np.unique(y))
ohm = np.zeros((y.shape[0], 10))
ohm[np.arange(y.shape[0]), y] = 1
y = np.reshape(y, (1, y.shape[0]))
X_train, X_test, y_train, y_test = train_test_split(X, ohm, test_size=0.30, random_state=42, stratify=ohm)


X_train = X_train.T
X_train = X_train/ 255
X_test = X_test.T
X_test = X_test/ 255
y_train = y_train.T
y_test = y_test.T
print(y_train.shape)
print(X_train.shape)
print(X_test.shape)

is_zeros = X.T[:, y[0,:] == 0][:, 0:2]
is_threes = X.T[:, y[0,:] == 3][:, 0:2]
is_fives = X.T[:, y[0,:] == 5][:, 0:2]
is_eights = X.T[:, y[0,:] == 8][:, 0:2]

is_zeros = np.reshape(is_zeros, (is_zeros.shape[0], 2))
is_threes = np.reshape(is_threes, (is_threes.shape[0], 2))
is_fives = np.reshape(is_fives, (is_fives.shape[0], 2))
is_eights = np.reshape(is_eights, (is_eights.shape[0], 2))

optimizer = nnet.Momentum(learning_rate=0.05)
regulizer = nnet.L2Regularization(lamda=0.05)
loss_func = nnet.SoftmaxCrossEntropyLoss()

newNet = nnet.Net(regularization=regulizer, optimizer=optimizer, cost_function=loss_func)

input = newNet.input_placeholder(shape=(784, None))
# Just showing all activations
hidden1 = newNet.dense(input, numOfUnits=100, activation=nnet.Relu())
hidden2 = newNet.dense(hidden1, numOfUnits=80, activation=nnet.Relu())
hidden3 = newNet.dense(hidden2, numOfUnits=50, activation=nnet.Relu())
hidden4 = newNet.dense(hidden3, numOfUnits=30, activation=nnet.Relu())
hidden5 = newNet.dense(hidden4, numOfUnits=20, activation=nnet.Relu())
hidden6 = newNet.dense(hidden5, numOfUnits=10, activation=nnet.Sigmoid())
output = newNet.dense(hidden6, numOfUnits=10, activation=nnet.Softmax())

num_epochs =500

costs = []

for epoch in range(num_epochs):
    _, loss = newNet.train(X_train, y_train)
    costs.append(loss)
    if( epoch % 10 == 0):
        pred_y_zeros = newNet.predict(is_zeros)
        print("Zeros: ",pred_y_zeros.T, " Predicted Values: ", np.argmax(pred_y_zeros, axis=0))
        pred_y_threes = newNet.predict(is_threes)
        print("Threes: ", pred_y_threes.T, " Predicted Values: ", np.argmax(pred_y_threes, axis=0))
        pred_y_fives = newNet.predict(is_fives)
        print("Fives: ", pred_y_fives.T, " Predicted Values: ", np.argmax(pred_y_fives, axis=0))
        pred_y_eights = newNet.predict(is_eights)
        print("Eights: ", pred_y_eights.T, " Predicted Values: ", np.argmax(pred_y_eights, axis=0))

        print("------ Epoch: ", epoch, " ------")
        print("Softmax entropy Loss: ", loss)
        print("----------------------------")



pred_y = newNet.predict(X_test)
pred_y = np.argmax(pred_y, axis=0)
y_test = np.argmax(y_test, axis=0)
print(pred_y[0:10])
print(y_test[0:10])
print("Shape Of Predictions: \t", pred_y.shape)
incorrect_X = np.not_equal(pred_y, y_test)
unique, counts = np.unique(incorrect_X, return_counts=True)
print("Predictions: \t", dict(zip(unique, counts)))
print("Precision score: \t", precision_score(y_test, pred_y))
print("Recall score: \t", recall_score(y_test, pred_y))
print("F1 score: \t", f1_score(y_test, pred_y))

# for i in range(incorrect_X.shape[0]):
#     if(i % 100 == 0):
#         d =0
#         #plt.figure(i)
#         #plt.imshow(incorrect_X[i].reshape(28,28))

# make an agg figure
fig, ax = plt.subplots()
ax.plot(costs)
ax.set_title('Loss Over Epochs')
fig.canvas.draw()
# grab the pixel buffer and dump it into a numpy array
X = np.array(fig.canvas.renderer._renderer)

# now display the array X as an Axes in a new figure
plt.show()

