#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Activation Functions

# ### Sigmoid
# * Pros
#     * Derivative is simple: z * (1 - z)
#     * For artificial networks it is preferable
# * Cons
#     * Not zero centered
#     * Saturate and kill gradients at either 0 or 1
#     * Computationally expensive because of exp()
#     * Not generally prefered in deep neural networks
#     

# In[ ]:


def sigmoid(z, prime = None):
    sigmoid = lambda t: 1.0/(1.0 + np.exp(-t))
    sigmoid_func = np.vectorize(sigmoid)
    if not prime:
        return sigmoid_func(z)
    prime = lambda t: sigmoid_func(t) * (sigmoid_func(t) + -1)
    prime_func = np.vectorize(prime)
    return prime_func(z)


# ### Rectified Linear Unit
# * Pros
#     * Computationally efficient
#     * Acceleration is quick
#     * 
# * Cons
#     * Can irreversibly die during training
#     * Not zero centered
#     * Very popular on deep neural networks

# In[ ]:


def relu(z, prime = None):
    relu = lambda t: t if t >= 0.0 else 0.0
    relu_func = np.vectorize(relu)
    if not prime:
        return relu_func(z)
    prime = lambda t: 1.0 if t >= 0.0 else 0.0
    prime_vectorized = np.vectorize(prime)
    return prime_vectorized(z)


# ### Linear
# * Pros
#     * Can be used for real valued outputs(no restriction)
# * Cons
#     * Cannot be used to make complex networks

# In[ ]:


def linear(z, prime = None):
    if not prime:
        return z
    return 1


# ### Hyperbolic Tangent
# * Pros
#     * Zero centered
# * Cons
#     * Kills gradient at -1 or 1

# In[ ]:


def tanh(z, prime = None):
    if not prime:
        return np.tanh(z)
    x = 1.0 - np.tanh(z)**2
    return x


# ### Leaky Relu
# * Pros
#     * Never saturate
#     * Computationally efficient
#     * Acceleration is quick
#     * Never die
# * Cons
#     * Relu is generraly more preferrable, but if you have lots of negative valued data, then you may use leaky Relu.

# In[ ]:


def leaky_relu(z, prime = None):
    relu = lambda t: t if t > 0.0 else 0.01 * t
    relu_func = np.vectorize(relu)
    if not prime:
        return relu_func(z)
    prime = lambda t: 1.0 if t > 0.0 else 0.01
    prime_func = np.vectorize(prime)
    return prime_func(z)


# ## Loss Functions

# * L1 loss = | predicted - target |
# * L2 loss = (predicted - target)^2

# In[ ]:


def L1(z, target = None, action = None): # Absolute error
    if not action:
        return z
    elif action == "prime":
        prime = lambda y, t: -1.0 if y < t else 1.0
        prime_vectorized = np.vectorize(prime)
        return prime_vectorized(z, target)
    elif action == "loss":
        return np.sum(np.absolute(z - target))
def L2(z, target = None, action = None): # Squared error
    if not action:
        return z
    elif action == "prime":
        return z - target
    elif action == "loss":
        return np.sum(np.square(z - target))


# ## Network Components

# ### Layer Class
# Layer class will be used to initialize hidden layers & output layer in the network.
#     
# * The variable: "weight_matrix" is a [node size X input size] numpy array. Every node has number of weights that is equal to input size(fully connected layer).
# * The method  : "forward_feed" returnes the output of a layer by using the input_matrix.
# * The method  : "derivative" used to calculate derivative of a layer by using the previous layers output and next layers derivative.
# * The method  : "derivative_out" is same as derivative but only used for outlayer.
# * The method  : "update_weights" is used to update the weight of a layer by using the derivative_matrix calculated by the method "derivative or derivative_out" and learning rate.

# In[ ]:


class Layer:
    def __init__(self, height, width, activation):
        if activation == relu or activation == leaky_relu:
            self.weight_matrix = np.random.rand(height, width) * np.sqrt(2/width)
        elif activation == tanh:
            self.weight_matrix = np.random.rand(height, width) * np.sqrt(1/width)
        else:
            self.weight_matrix = np.random.rand(height,width)
        self.bias_matrix = np.zeros((height, width))
        self.activation = activation
    def forward_feed(self, input_matrix):
        return self.activation(
            np.dot(self.weight_matrix, input_matrix)) + np.sum(self.bias_matrix, axis=1)
    def derivative(self, left_input, right_input):
        return (self.weight_matrix.T * self.activation(np.sum(right_input, axis=0), 1)).T * left_input.T
    def derivative_out(self, left_input, right_input, target_input):
        return (self.weight_matrix.T * self.activation(right_input, target_input, "prime").T).T * left_input.T
    def update_weights(self, derivative_matrix, learning_rate):
        self.weight_matrix -= learning_rate * derivative_matrix
        self.bias_matrix -= learning_rate * derivative_matrix


# ### Network Class
# * The method: "init" method is used to initalize a network. You should enter input_size and output_size that matches your data
# * The method: "add_out_layer" is used to add the output layer. You can chose either L1 or L2 loss.
# * The method: "add_layer" is used to add an hidden layer to network provided the node size(height). You can use any activation function provided above.
# * The method: "predict" is used to run the network for a given input. Returns the predicted out. If the argument: "return_all" is true then returns [input + hidden outs + out] as a list
# * The method: "back_prop" is used to run back_propagation algorithm provided the input and desired output. Automaticaly updates the weights.
# * The method: "train" is just calls back_prop for every item in lists "input_list" and "target_list"
# * The method: "batch_train" is same as train, but only updates the best loss.
# * The method: "loss" returns the loss of a input and target pair.

# In[ ]:


class Network:
    # Initialization function
    def __init__(self, input_size, output_size):
        self.layers = []
        self.input_size = input_size
        self.output_size = output_size
        
    # Add Layers
    def add_out_layer(self, loss_function = L1):
        l = Layer(self.output_size, self.layers[-1].weight_matrix.shape[0], loss_function)
        self.layers.append(l)
        
    def add_layer(self, height, activation = relu):
        if not self.layers:
            width = self.input_size
        else:
            width = self.layers[-1].weight_matrix.shape[0]
        input_height = self.layers
        l = Layer(height, width, activation)
        self.layers.append(l)
    # Predict an input
    def predict(self, input_matrix, return_all = False):
        last_out = input_matrix
        out_list = [input_matrix]
        for layer in self.layers:
            last_out = layer.forward_feed(last_out)
            if return_all:
                out_list.append(last_out)
        if return_all:
            return out_list
        return last_out
    # back_prop an (input, target) pair and update weights
    def back_prop(self, input_matrix, target_matrix):
        out_list = self.predict(input_matrix, 1)
        derivative_matrix = self.layers[-1].derivative_out(out_list[-2], out_list[-1], target_matrix)
        self.layers[-1].update_weights(derivative_matrix, self.learning_rate)
        i = -3
        for layer in reversed(self.layers[:-1]):
            derivative_matrix = layer.derivative(out_list[i], derivative_matrix)
            layer.update_weights(derivative_matrix, self.learning_rate)
            i -= 1
        return self.layers[-1].activation(out_list[-1], target_matrix, "loss")
    def train(self, input_list, target_list, learning_rate = 0.1):
        batch_loss = 0.0
        self.learning_rate = learning_rate
        for input_matrix, target_matrix in zip(input_list, target_list):
            batch_loss += self.back_prop(input_matrix, target_matrix)
        return batch_loss
    def batch_train(self, input_list, target_list, learning_rate = 0.1):
        self.learning_rate = learning_rate
        loss_before = self.loss(input_list, target_list)
        best_layer_model = self.layers
        best_loss = self.layers[-1].activation(self.predict(input_list[0]), target_list[0], "loss")
        
        for input_matrix, target_matrix in zip(input_list[1:], target_list[1:]):
            current_loss = self.back_prop(input_matrix, target_matrix)
            if best_loss < current_loss:
                self.layers = best_layer_model
            else:
                best_loss = current_loss
                best_layer_model = self.layers
            return loss_before - self.loss(input_list, target_list)
    def loss(self, input_list, target_list):
        loss = 0.0
        for x, y in zip(input_list, target_list):
            loss += self.layers[-1].activation(self.predict(x), y, "loss")
        return loss


# ## Storing Training Data to Dataframes

# In[ ]:


test_data = pd.read_csv("../input/testSimple.csv")
train_data = pd.read_csv("../input/trainSimple.csv")


# ## Initializing the Network
# 6 is the input size and 2 is the output size

# In[ ]:


n = Network(6, 2)


# ## Adding layers
# You can layers as you like but dont forget to add an out layer(L1 or L2 loss may be used. The format is : add_layer( node size, activation function)

# In[ ]:


n.add_layer(24, leaky_relu)
n.add_layer(24, tanh)
n.add_out_layer(L1)


# ## Training the Network
# What that code does is that;
#     
# * Runs train algorithm on whole data for 20 times(epoch).
# * Every epoch data is a randomly shuffled train data.
# * For every 12 [input, target] pair(batches), only the best loss provided by the backpropagation algorithm is used to update weights.

# In[ ]:


from sklearn.utils import shuffle
learning_rate = 0.001
batch_size = 12
epoch = 20
previous_total_loss = n.loss(train_data.values[:, :6], train_data.values[:, 6:])
for i in range(epoch):
    x = shuffle(train_data.values)
    improvement = 0
    for batch in np.array_split(x, len(x)/batch_size):
        improvement += n.batch_train(batch[:, :6], batch[:, 6:], learning_rate)
    epoch_loss = n.loss(train_data.values[:, :6], train_data.values[:, 6:])
    print("MAE loss for epoch:" + str(i) + " = " + str(epoch_loss))
    print("Improvement = " + str(improvement))
current_total_loss =n.loss(train_data.values[:, :6], train_data.values[:, 6:])
print("Current Total Loss = " + str(current_total_loss))
print("Total Improvement(MAE) = " + str(previous_total_loss - current_total_loss))


# ## Analyzing Predictions on 10 random data

# In[ ]:


sample_data = shuffle(train_data)
train_inputs = sample_data.iloc[:, :6].values
train_targets = sample_data.iloc[:, 6:].values
data_number = 0
for train_input, train_target in zip(train_inputs[:10], train_targets[:10]):
    data_number += 1
    print("Sample Data " + str(data_number))
    print("Prediction:")
    prediction = n.predict(train_input)
    print(prediction)
    print("Target:")
    print(train_target)
    print("L1 loss:")
    print(L1(prediction, train_target, "loss"))


# ## Lets use our trained network on test data

# In[ ]:


test = pd.read_csv("../input/testSimple.csv")


# In[ ]:


test.tail()


# In[ ]:


predicted_list = []
for row in test.values:
    predicted_list.append(np.append(row[0], n.predict(row[1:])))
predicted_df = pd.DataFrame(predicted_list, columns = ['ID', 'A', 'B'])
predicted_df['ID'] = predicted_df['ID'].astype(int)


# In[ ]:


predicted_df.head()


# ## Lets download the predictions on test data

# In[ ]:


predicted_df.to_csv('submission.csv', index=False)

