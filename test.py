import nnet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

def predictionReshape(probas, m):
    p = np.zeros((1,m))
    less_than_half = (probas[0,:] <= 0.5).astype(np.int)
    greater_than_half = (probas[0,:] > .5).astype(np.int)
    p_temp = np.multiply(less_than_half, 0)
    p[0] = np.add(p_temp, greater_than_half)
    return p


digits = fetch_mldata("MNIST original")

X, y = digits["data"],digits["target"]
print(np.unique(y))
y_5 = (y == 5).astype(np.float32)
print(np.unique(y_5))
X_train, X_test, y_train, y_test = train_test_split(X, y_5, test_size=0.30, random_state=42, stratify=y_5)

X_train = X_train.T
X_train = X_train/ 255
X_test = X_test.T
X_test = X_test/ 255
y_train = np.reshape(y_train, (1, y_train.shape[0]))
y_test = np.reshape(y_test, (1, y_test.shape[0]))
print(y_train.shape)
print(X_train.shape)
print(X_test.shape)

is_five = X_test[:, y_test[0,:] == 1][:, 0:5]
is_not_five = X_test[:, y_test[0,:] != 1][:, 0:5]
plt.figure("FIVE")
#plt.imshow(is_five.reshape(28,28))
plt.figure("NOT FIVE")
#plt.imshow(is_not_five.reshape(28,28))
is_five = np.reshape(is_five, (is_five.shape[0], 5))
is_not_five = np.reshape(is_not_five, (is_not_five.shape[0], 5))

optimizer = nnet.Momentum(learning_rate=1)
regulizer = nnet.L2Regularization(lamda=0.0001)
loss_func = nnet.CrossEntropyLoss()

newNet = nnet.Net(regularization=regulizer, optimizer=optimizer, cost_function=loss_func)

input = newNet.input_placeholder(shape=(784, None))
# Just showing all activations
hidden1 = newNet.dense(input, numOfUnits=50, activation=nnet.Relu())
hidden2 = newNet.dense(hidden1, numOfUnits=30, activation=nnet.Relu())
hidden3 = newNet.dense(hidden2, numOfUnits=10, activation=nnet.Relu())
hidden4 = newNet.dense(hidden3, numOfUnits=4, activation=nnet.Relu())
output = newNet.dense(hidden4, numOfUnits=1, activation=nnet.Sigmoid())

num_epochs = 5000

costs = []

for epoch in range(num_epochs):
    _, loss = newNet.train(X_train, y_train)
    costs.append(loss)
    if( epoch % 5 == 0):
        pred_y_V = newNet.predict(is_five)
        print(pred_y_V)
        pred_y_V = predictionReshape(pred_y_V, 5)
        print(pred_y_V)
        print("---------------")
        pred_y_V2 = newNet.predict(is_not_five)
        print(pred_y_V2)
        pred_y_V2 = predictionReshape(pred_y_V2, 5)
        print(pred_y_V2)
        print("------ Epoch: ", epoch, " ------")
        print("Corss entropy Loss: ", loss)
        print("----------------------------")



pred_y = newNet.predict(X_test)
print(pred_y.shape)
pred_y = predictionReshape(pred_y, X_test.shape[1])
print(y_test.shape)
print("Shape Of Predictions: \t", pred_y.shape)
incorrect_X = np.not_equal(pred_y, y_test)
unique, counts = np.unique(incorrect_X, return_counts=True)
print("Predictions: \t", dict(zip(unique, counts)))
print("Precision score: \t", precision_score(y_test[0,:], pred_y[0,:]))
print("Recall score: \t", recall_score(y_test[0,:], pred_y[0,:]))
print("F1 score: \t", f1_score(y_test[0,:], pred_y[0,:]))

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

