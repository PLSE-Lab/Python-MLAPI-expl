#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 


# ![](https://cdn-images-1.medium.com/max/1600/1*sX6T0Y4aa3ARh7IBS_sdqw.png)

# In[ ]:


#Helper Functions

import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# derivative of sigmoid
# sigmoid(y) * (1.0 - sigmoid(y))
# the way we use this y is already sigmoided
def sigmoid_derivative(y):
    return y * (1.0 - y) 


# ![](https://cdn-images-1.medium.com/max/1600/1*7zxb2lfWWKaVxnmq2o69Mw.png)

# In[ ]:


class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


# In[ ]:


X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0],[1],[1],[0]])
nn = NeuralNetwork(X,y)


# In[ ]:


#epochs
for i in range(1000):
    nn.feedforward()
    nn.backprop()


# In[ ]:


nn.weights1


# In[ ]:


nn.weights2


# In[ ]:


for i in range(y.shape[0]):
    print(nn.output[i],y[i])

