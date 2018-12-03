# DenseNeuralNet
Simple implementation of a Neural Network framework. Only meant for me to reinforce some of the knowledge I have gained about Neural Networks.

### Topic Of Interest
- Weight initialization of layers(He, Xavier, Reduced Standard Normal)
- Dropout layers
- Optimizers(Gradient Descent, Momentum, RMSProp, Adam)
- Activation functions(Sigmoid, Softmax, ReLU, Leaky ReLU, ELU, Tanh)
- Loss functions(Cross Entropy Loss, Softmax Cross Entropy, MSE<Not working>)

Used for Classification, to see examples see 1_Basic_Example.ipynb, and 2_MNIST_Example.ipynb

### Basic Usage
1. Import and create neural network class
    ```
       import nnet
       neural_net = nnet.Net(regularization=L2Regularization(lamda=0), 
                    optimizer=GradientDescent(learning_rate=0.01),
                    cost_function=CrossEntropyLoss()
                    )
    ```
2. Add layers, starting with the input -> hidden layers -> output
    ```
        input = newNet.input_placeholder(shape=("Number Of Input Features", None))
        hidden_layer_1 = newNet.dense(perviousLayer=input, 
                                    numOfUnits=20, 
                                    initilization="he", 
                                    activation=Sigmoid(), 
                                    keep_prob=1.0
                                    )
        # As many layers as you want, ...
        output = newNet.dense(hidden_layer_x, 
                            numOfUnits=10, # Number of classes
                            initilization="he", 
                            activation=Sigmoid(), 
                             keep_prob=1.0
                            )
    ```
3. Train network
    ```
        # in a for loop
        layers, loss = newNet.train(X, Y, _check_gradients = False)
    ```
4. Predict Results
    ```
        pred_y = newNet.predict(X)
    ```
    
**NB: This has been tested, on simple data for classification tasks, which seems to work for binary and multi-class. I did write an MSE loss for regression tasks, but it was not tested or debugged, nor am I planning to. That being said, I won't recommend using this for production applications, I made this to reinforce some of my knowledge, didn't put too much time in it and it will not be maintained. Use Tensorflow, or pytorch**