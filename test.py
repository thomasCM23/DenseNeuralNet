import nnet
import numpy as np
newNet = nnet.Net()

input = newNet.input_placeholder(shape=(2, None))

hidden1 = newNet.dense(input, numOfUnits=4, activation=nnet.Relu())

output = newNet.dense(hidden1, numOfUnits=1, activation=nnet.Sigmoid())

d = newNet.train(np.array([[1.0,2.0],[1.0,3.0],[1.0,6.0],[3.0,2.0],[4.0,2.0]]).T, np.array([1.0,1.0,1.0,0.0,0.0]))

print(d)

