#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import numpy as np
import pylab
import random

def sigmoid(x) :
    return 1/(1+np.exp(-x))

def derivate_sigmoid(x) :
    return sigmoid(x) * (1-sigmoid(x))

def single_node(inputs, weights) :
    v = np.dot(inputs , weigths)
    return v

def error(target,output) :
    return 0.5 * (target -output) ** 2

def activate(val) :
    return sigmoid(val)

def gradient_output(input,target,output) :
    return derivate_sigmoid(input) * (target- output)

def gradient_hidden(input,gradients,weights) :
    sum = 0
    for i in range(len(gradients)) :
        sum +=gradients[i] * weights[i]
    return derivate_sigmoid(input) * sum

def update_weight(weight,momentum,learning_rate,gradient,output) :
    return weight + momentum *weight + learning_rate * gradient * output

def initiate_weight(input_size, hidden_layer, nb_nodes):
    weights = []
    weights_input_hidden = []
    weights_hidden_hidden = []
    weights_hidden_output = []
    
    # Random weight from input layer to hidden layer
    for i in range(0, nb_nodes):
        nb_nodes_i = []
        for j in range(0, input_size):
            weight = random.uniform(0.0, 1.0)
            nb_nodes_i.append(weight)
        weights_input_hidden.append(nb_node_i)
    weights.append(weights_input_hidden)
    
    # Random weight from hidden layer to hidden layer
    for i in range(1, hidden_layer):
        weights_hidden_hidden = []
        for j in range(0, nb_nodes):
            nb_nodes_i = []
            for k in range (0, nb_nodes):
                weight = random.uniform(0.0, 1.0)
                nb_nodes_i.append(weight)
            weights_hidden_hidden.append(nb_nodes_i)
        weights.append(weights_hidden_hidden)

    # Random weight from hidden layer to output layer
    for i in range(0, nb_nodes):
        weight = random.uniform(0.0, 1.0)
        weights_hidden_output.append(weight)
    weights.append(weights_hidden_output)
    
    return weights

