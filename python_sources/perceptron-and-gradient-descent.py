#!/usr/bin/env python
# coding: utf-8

# __What does a neuron do ?__
# 
# It calculates the weighted sum and adds bias to it and decides whether to activate it or not. if we see the equation of a neuron it can be written as 
#                                                                                            
#                                              Y = sum(weight * input) + bias
#     
# value of y can be ranging from -inf to +inf and with this range we do not know how to decide when to activate the neuron. Activation functions help us to decide whether to activate or not by mapping the value in that range to various ranges and decide accordingly

# In[ ]:


import numpy as np


# In[ ]:


def linear_neuron(x,weight):
        prediction = (x*weight)
        return prediction


# In[ ]:


weight = 0.2
x = [1.0,2.0,3.0] #first feature with three samples
linear_neuron(x[0],weight) #sending the first sample as input


# In[ ]:


def with_dot(x,weight):
    x = np.array(x)
    weight = np.array(weight)
    prediction = x.dot(weight)
    return prediction


# In[ ]:


def for_three(x,weight):
    result = [0.0,0.0,0.0]
    for i in range(len(x)):
        result[i] = linear_neuron(x[i],weight)
    return result


# # Multiple Inputs

# In[ ]:


def w_sum(a,b):
    assert(len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output

weights = [0.1, 0.2, 0] 
    
def neural_network(input, weights):
    pred = w_sum(input,weights)
    return pred

# This dataset is the current
# status at the beginning of
# each game for the first 4 games
# in a season.

# toes = current number of toes
# wlrec = current games won (percent)
# nfans = fan count (in millions)

toes =  [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

# Input corresponds to every entry
# for the first game of the season.

input = [toes[0],wlrec[0],nfans[0]]
pred = neural_network(input,weights)

print(pred)


# In[ ]:


#Using Numpy

import numpy as np
weights = np.array([0.1, 0.2, 0])
def neural_network(input, weights):
    pred = input.dot(weights)
    return pred
    
toes =  np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

# Input corresponds to every entry
# for the first game of the season.

input = np.array([toes[0],wlrec[0],nfans[0]])
pred = neural_network(input,weights)

print(pred)


# # Multiple Outputs

# In[ ]:


def ele_mul(number,vector):
    output = [0,0,0]
    assert(len(output) == len(vector))
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output

weights = [0.3, 0.2, 0.9] 

def neural_network(input, weights):
    pred = ele_mul(input,weights)
    return pred
    
wlrec = [0.65, 0.8, 0.8, 0.9]
input = wlrec[0]
pred = neural_network(input,weights)

print(pred)


# In[ ]:


weights = [ [0.1, 0.1, -0.3], 
            [0.1, 0.2, 0.0], 
            [0.0, 1.3, 0.1] ] 

def w_sum(a,b):
    assert(len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output

def vect_mat_mul(vect,matrix):
    assert(len(vect) == len(matrix))
    output = [0,0,0]
    for i in range(len(vect)):
        output[i] = w_sum(vect,matrix[i])
    return output

def neural_network(input, weights):
    pred = vect_mat_mul(input,weights)
    return pred

t =  [8.5, 9.5, 9.9, 9.0]
w = [0.65,0.8, 0.8, 0.9]
n = [1.2, 1.3, 0.5, 1.0]

input = [t[0],w[0],n[0]]
pred = neural_network(input,weights)

print(pred)


# 
# # Perceptron
# 
# <img src="https://cdn-images-1.medium.com/max/1600/1*n6sJ4yZQzwKL9wnF5wnVNg.png" width="400px">

# In[ ]:


from random import choice 
from numpy import array, dot, random 
unit_step = lambda x: 0 if x < 0 else 1 
training_data = [(array([0,0,1]), 0), 
                 (array([0,1,1]), 1), 
                 (array([1,0,1]), 1), 
                 (array([1,1,1]), 1), ] 
w = random.rand(3)
print(w)
errors = [] 
 
n = 50 
for i in range(n): 
    x, expected = choice(training_data) 
    result = dot(w, x) 
    error = expected - unit_step(result) 
    #print(error)
    errors.append(error)
    w +=  error * x 
for x, _ in training_data: 
    result = dot(x, w) 
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))


# In[ ]:


#How are error and weight varying?


# In[ ]:


#what is bias and why do we need it? 


# In[ ]:


# try for the data "and"
"""
x1 x2 y
0  0  0
1  0  0
0  1  0
1  1  1
"""

#try for the data "xor"
"""
x1 x2 y
0  0  0
1  0  1
0  1  1
1  1  0
"""


# # Problems with Perceptrons 
# 
# convergence is one of the biggest problems of the perceptron. Frank Rosenblatt proofed mathematically that the perceptron learning rule converges if the two classes can be separated by linear hyperplane, but problems arise if the classes cannot be separated perfectly by a linear classifier.
# 
# since one or more samples will always be misclassified in every epoch so that the learning rule never stops updating the weights.
# 
# Also, it stops updating the weights as soon as all samples are classified correctly. Which doesn't ensure the it has generalized well enough.

# Check out this [link](https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html#gradient-descent) for Gradient Descent 

# In[ ]:


# Adaline Gradient Descent

# The ADAptive LInear NEuron (Adaline) is similar to the Perceptron, 
# except that it defines a cost function based on the soft output and an optimization problem.
# We can therefore leverage various optimization techniques to train Adaline in a more theoretic grounded manner. 
# Let's implement the Adaline using the batch gradient descent (GD) algorithm:

import numpy as np

class AdalineGD(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.epochs):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


# In[ ]:


X = np.array([[1,0],[1,1],[0,1],[0,0]])
y = np.array([1,1,1,0])
ada = AdalineGD()
ada.train(X, y)


# In[ ]:


ada.predict([0,0])


# In[ ]:




