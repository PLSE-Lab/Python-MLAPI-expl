#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# When I started studying Neural Networks, I approached the challenge the same way I was approaching other learning endeavors: First **I lay down the theoretical foundations** by reading books and formal writing on the topic. Then **I implement the topics myself** to cement my theoretical understanding, and finally **I get hands-on and apply what I learned** to perform tasks and gain experience. When I set out to take the second step (implementing a Neural Network from scratch) I hit a brick wall. I was searching for tutorials and guides, but I found all the material inadequate for what I wanted to accomplish. A lot of resources were focusing too much on the theoretical background of the algorithm and put forth little in terms of explaining how their implementation worked. Others went into detail about the workings of their code, but the code was too simple to be used in more general problems. I wanted an easy-to-read tutorial where the result would be a multi-layered neural network that could be used for non-trivial problems.
# 
# In the past couple of years this has been amended with some excellent articles shared online, but I want to add to this collection my own work for the sake of my younger self.
# 
# So, if you want a tutorial that will not delve deep into the mathematical background neural networks and will instead focus on a simple-enough-but-not-barebones implementation, you have come to the right place!
# 
# *Note: As should be obvious, I assume you are already familiar with neural networks. Here I simply put this knowledge into actual code. This is the second phase of my 3-phase learning schedule I outlined above.*

# ## Data
# 
# Before the actual implementation, we need to read and process our data. For the purposes of this tutorial, we will use the famous Iris Species Dataset. You can find more information on the Dataset Page right here on [Kaggle](https://www.kaggle.com/uciml/iris), In short, the dataset contains data for **three** species of a flower and we have **four** pieces of data for each sample (sepal length/width and petal length/width).
# 
# The two libraries we will mainly use are `numpy` for the mathematical operations and `pandas` for reading the dataset (we will also use another library later on). Let's import them!

# In[ ]:


import numpy as np
import pandas as pd


# Next we will read the dataset using `pandas` and shuffle it. Shuffling a dataset, when the order of the samples does not matter, is usually for the best. I will not go into detail, but a shuffled dataset is better both for separating the dataset into train/test/validation and for avoiding overfitting.

# In[ ]:


iris = pd.read_csv("../input/Iris.csv")
iris = iris.sample(frac=1).reset_index(drop=True) # Shuffle


# We need to grab the data (the information on each sample) from the `pandas` array and put it into a nice `numpy` one.

# In[ ]:


X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
X = np.array(X)
X[:5]


# The above will be fed in our neural network for training. Only thing that's missing now is to format the class of each item. Namely, we need to convert classes from categorical ('Setosa', 'Versicolor', 'Virginica') to numerical (0, 1, 2) and then to one-hot encoded ([1, 0, 0], [0, 1, 0], [0, 0, 1]). A class in one-hot encoded form is an array of 0s (a 0 for each different class) with one element equal to 1 in the index of the class (if the value of the class is 3, index 3 of the one-hot array will be equal to 1).
# 
# The above can be done easily and without hassle using the `OneHotEncoder` function from the `sklearn` library. We will simply take the class information (under the 'Species' column) of each sample and convert it using the imported function.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)

Y = iris.Species
Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))
Y[:5]


# Next we will split our dataset into train/validation/test using `sklearn`. Initially the data will be split into train/test and then the training data will be further split into train/validation.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)


# ## Implementation
# 
# In this tutorial, we are going to build a simple neural network that supports multiple layers and validation. The main function is `NeuralNetwork`, which will train the network for the specified number of epochs. At first, the weights of the network will get randomly initialized by `InitializeWeights`. Then, in each epoch, the weights will be updated by `Train` and finally, every 20 epochs accuracy both for the training and validation sets will be printed by the `Accuracy` function. As input the function receives the following:
# 
# * `X_train`, `Y_train`: The training data and target values.
# * `X_val`, `Y_val`: The validation data and target values. These are optional parameters.
# * `epochs`: Number of epochs. Defaults at 10.
# * `nodes`: A list of integers. Each integer denotes the number of nodes in each layer. The length of this list denotes the number of layers. That is, each integer in this list corresponds to the number of nodes in each layer.
# * `lr`: The learning rate of the back-propagation training algorithm. Defaults at 0.15.

# In[ ]:


def NeuralNetwork(X_train, Y_train, X_val=None, Y_val=None, epochs=10, nodes=[], lr=0.15):
    hidden_layers = len(nodes) - 1
    weights = InitializeWeights(nodes)

    for epoch in range(1, epochs+1):
        weights = Train(X_train, Y_train, lr, weights)

        if(epoch % 20 == 0):
            print("Epoch {}".format(epoch))
            print("Training Accuracy:{}".format(Accuracy(X_train, Y_train, weights)))
            if X_val.any():
                print("Validation Accuracy:{}".format(Accuracy(X_val, Y_val, weights)))
            
    return weights


# The weights of the network are initialized randomly in the range [-1, 1] by `InitializeWeights`. This function takes as input `nodes` and returns a multi-dimensional array, `weights`. Each element in the `weights` list represents a hidden layer and holds the weights of connections from the previous layer (including the bias) to the current layer. So, element `i` in `weights` holds the weights of the connections from layer `i-1` to layer `i`. Note that the input layer has no incoming connections so it is not present in `weights`.
# 
# For example, let's say we have four features (as is the case with the Iris dataset) and the hidden layers have 5, 10 and 3 (for the output, one for each class) nodes. Thus, `nodes == [4, 5, 10, 3]`  Then, the connections between the input layer and the first hidden layer will be (4+1)\*5 = 25. After augmenting the input with the bias (in this case the bias has a constant value of 1), the input layer has 5 nodes. By fully connecting this layer to the next (each node in the input layer is connected will every node of the hidden layer), we get that the total number of connections is 25. Similarly, we get that the connections between the first hidden layer and the second one will be (5+1)\*10 = 60 and between the second hidden layer with the output we have (10+1)\*3 = 33 connections.
# 
# In the implementation, `numpy` is used to generate a random number in the `[-1, 1]` range for each connection.

# In[ ]:


def InitializeWeights(nodes):
    """Initialize weights with random values in [-1, 1] (including bias)"""
    layers, weights = len(nodes), []
    
    for i in range(1, layers):
        w = [[np.random.uniform(-1, 1) for k in range(nodes[i-1] + 1)]
              for j in range(nodes[i])]
        weights.append(np.matrix(w))
    
    return weights


# With the weights of the network at hand, we want to continuously adjust them across the epochs so that (hopefully) our network becomes more accurate. The training of the weights is accomplished via the popular (Forward) Back-Propagation algorithm. In this technique, the input first passes through the whole network and the output is calculated. Then, according to the error of this output, the weights of the network are updated from last to first. The error is propagated *backwards*, hence the name of the titular algorithm. Let's get into more detail about these two steps:
# 
# **Forward Propagation:**
# 
# * Each layer receives an input and computes an output. The output is computed by first calculating the dot product between the input and the weights of the layer and then passing this dot product through an activation function (in this case, the sigmoid function).
# * The output of each layer is the input of the next.
# * The input of the first layer is the feature vector.
# * The output of the final layer is the prediction of the network.

# In[ ]:


def ForwardPropagation(x, weights, layers):
    activations, layer_input = [x], x
    for j in range(layers):
        activation = Sigmoid(np.dot(layer_input, weights[j].T))
        activations.append(activation)
        layer_input = np.append(1, activation) # Augment with bias
    
    return activations


# **Backward Propagation:**
# 
# * Calculate error at final output.
# * Propagate error backwards through the layers and perform corrections.
#     * Calculate Delta: Back-propagated error of current layer *times* Sigmoid derivation of current layer activation.
#     * Update Weights between current layer and previous layer: Multiply delta with activation of previous layer and learning rate, and add this product to weights of previous layer.
#     * Calculate error for current layer. Remove the bias from the weights of the previous layer and multiply the result with delta to get error.
# 
# It is easier if we think of updates as operating on the weights between two layers: $a$ | $W$ | $b$. When updating $W$, we are given the back-propagated error of $b$. Then, we calculate the delta, which is the error times the sigmoid derivative of $b$'s activation. Then, we multiply the delta with the activation of $a$ and the learning rate and add it to $W$. Finally, we calculate the new error and propagate it backwards again.
# 
# This is an iterative process that goes from output to input. The first error in the process is the final output error.

# In[ ]:


def BackPropagation(y, activations, weights, layers):
    outputFinal = activations[-1]
    error = np.matrix(y - outputFinal) # Error at output
    
    for j in range(layers, 0, -1):
        currActivation = activations[j]
        
        if(j > 1):
            # Augment previous activation
            prevActivation = np.append(1, activations[j-1])
        else:
            # First hidden layer, prevActivation is input (without bias)
            prevActivation = activations[0]
        
        delta = np.multiply(error, SigmoidDerivative(currActivation))
        weights[j-1] += lr * np.multiply(delta.T, prevActivation)

        w = np.delete(weights[j-1], [0], axis=1) # Remove bias from weights
        error = np.dot(delta, w) # Calculate error for current layer
    
    return weights


# In our implementation we will pass each sample of our dataset through the network, performing first the forward pass and then the weight updating via the back-propagation algorithm. Finally, the newly calculated weights will be returned.

# In[ ]:


def Train(X, Y, lr, weights):
    layers = len(weights)
    for i in range(len(X)):
        x, y = X[i], Y[i]
        x = np.matrix(np.append(1, x)) # Augment feature vector
        
        activations = ForwardPropagation(x, weights, layers)
        weights = BackPropagation(y, activations, weights, layers)

    return weights


# Neural networks need an activation function to pass the dot product of each layer through to get the final output (as well as to get to get the delta in back-propagation). In this tutorial, we will use the sigmoid function and its derivative. Other activation functions are available, like the famous *ReLU*. Also, sometimes layers don't use the same activation function, and there are times where the output doesn't have an activation function at all (for example, in the case of regression).

# In[ ]:


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SigmoidDerivative(x):
    return np.multiply(x, 1-x)


# When we want to make a prediction for an item, we need to first pass it through the network. The output of the network (in the case of three different classes, as in the Iris problem) will be in the form `[x, y, z]` where `x, y, z` are real numbers in the range [0, 1]. The higher the value of an element, the more confident the network is that it is the correct class. We need to convert this output to the proper one-hot format we mentioned earlier. Thus, we will take the largest of the outputs and set the corresponding index to 1, while the rest are set to 0. This means the predicted class is the one the network is most confident in (ie. the greatest value).
# 
# So, a prediction involves the forward propagation and the conversion of the output to one-hot encoding, with the 1 denoting the predicted class.

# In[ ]:


def Predict(item, weights):
    layers = len(weights)
    item = np.append(1, item) # Augment feature vector
    
    ##_Forward Propagation_##
    activations = ForwardPropagation(item, weights, layers)
    
    outputFinal = activations[-1].A1
    index = FindMaxActivation(outputFinal)

    # Initialize prediction vector to zeros
    y = [0 for i in range(len(outputFinal))]
    y[index] = 1  # Set guessed class to 1

    return y # Return prediction vector


def FindMaxActivation(output):
    """Find max activation in output"""
    m, index = output[0], 0
    for i in range(1, len(output)):
        if(output[i] > m):
            m, index = output[i], i
    
    return index


# Finally, we need a way to evaluate our network. For this, we will write the `Accuracy` function which, given the computed weights, predicts the class of each object in its input and checks it against the actual class, returning the percentage of correct predictions.
# 
# Instead of the percentile accuracy, other accuracy metrics can be employed, but for this tutorial this simple method will do.

# In[ ]:


def Accuracy(X, Y, weights):
    """Run set through network, find overall accuracy"""
    correct = 0

    for i in range(len(X)):
        x, y = X[i], list(Y[i])
        guess = Predict(x, weights)

        if(y == guess):
            # Guessed correctly
            correct += 1

    return correct / len(X)


# We have now completed our implementation and we can check the results! Below we build a network by passing to the main function (`NeuralNetwork`) the training/validation sets, the number of epochs, the learning rate and the number of nodes in each layer.
# 
# During the training, after each 20th epoch, the accuracy of the network on the training and validation sets will be printed.

# In[ ]:


f = len(X[0]) # Number of features
o = len(Y[0]) # Number of outputs / classes

layers = [f, 5, 10, o] # Number of nodes in layers
lr, epochs = 0.15, 100

weights = NeuralNetwork(X_train, Y_train, X_val, Y_val, epochs=epochs, nodes=layers, lr=lr);


# For the grand finale, we will test the network against the testing dataset:

# In[ ]:


print("Testing Accuracy: {}".format(Accuracy(X_test, Y_test, weights)))


# And that is all. We built a neural network from scratch and trained it to predict Iris species. I hope this journey was as helpful and educational to you as it was for me when I was starting out.
# 
# Thanks for reading!
