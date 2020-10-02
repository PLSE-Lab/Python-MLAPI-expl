#!/usr/bin/env python
# coding: utf-8

# # *Welcome to FIO Labs*
# 
# ## More Tutorials are available in Video format on our [YouTube Channel - FIOLabs](https://www.youtube.com/channel/UC6Vn_nUJeJ7PrvsayHPVOag). Follow for more AI | ML | DL Tutorials.

# # Multi-Layer Perceptron
# 
# Our last tutorial on [Shallow Neural Network](https://www.kaggle.com/prashfio/shallow-neural-network-ai-tutorials-fio-labs) points out a major drawback, i.e., we cannot extract more features from a given set of data when only one hidden layer is used. So, the solution is to apply **Multi-Layer Perceptron** to the problem when we have more high level features to extract. The reason is that for real-world problems, we need neural networks which can take *n* number of inputs, have *n* number of hidden layers and also *n* number of nodes in each hidden layer. The same applies to output layer. So, we will code accordingly. A multi-layer perceptron is also called as **Deep Neural Network**.

# # Structure of Multi-Layer Perceptron
# 
# Let us first look at the structure of Multi-Layer Perceptron to get a better understanding on how it works internally.

# In[ ]:


from IPython.display import Image
import os
get_ipython().system("ls '../input/'")


# In[ ]:


Image('../input/Multilayerp.png')


# As you can see, the structure gets more and more complex when number of hidden layers and nodes are increased in each layer. This is the reason why it is computationally challenging to run few neural networks.
# 
# Let's go ahead and start coding.

# # 1. Initialize Neural Network
# 
# For a neural network, we need to have *number of inputs, number of hidden layers, number of nodes in each hidden layer, and number of outputs*. So, we need to define a function where we can just input these numbers and it would initialize the network with randomly assigned weights.

# In[ ]:


import numpy as np

def initialize_neural_network(n_inputs, n_hidden_layers, n_nodes_hidden, n_nodes_outputs):
    
    '''
    Creating a function to initialize neural network.
    n_inputs = Number of inputs
    n_hidden_layers = Number of hidden layers
    n_nodes_hidden = Number of nodes in each hidden layer
    n_nodes_outputs = Number of outputs.
    '''
    
    num_nodes_previous = n_inputs #Whatever outputs we get from previous layers will be the inputs for the next layers.
    network = {} #Let's initialize a network with empty dictionary. This will be used later to display/describe the whole network.
    
    #let's create a loop to randomly intitialize the weights and biases in each layer
    #we're adding 1 to every hidden layer so as to include the output layer.
    
    for layer in range(n_hidden_layers + 1):
        #let's define layer names.
        if layer == n_hidden_layers:
            layer_name = 'output' #name of our last layer
            n_nodes = n_nodes_outputs
        else:
            layer_name = 'layer_{}'.format(layer + 1) #This will iterate the number of hidden layers until you reach output layer
            n_nodes = n_nodes_hidden[layer]
            
        # let's intitialize weights and bias for each node
        
        network[layer_name] = {}
        for node in range(n_nodes):
            node_name = 'node_{}'.format(node+1) #define node names
            network[layer_name][node_name] = {
                'weights' : np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias' : np.around(np.random.uniform(size=1), decimals=2)
            }
            
        num_nodes_previous = n_nodes
            
    return network


# Now that we defined function to intialize the network, let's give some values to see what result we get when we call this function.

# In[ ]:


my_network = initialize_neural_network(4, 3, [3,2,3], 2) # 4 inputs, 3 hidden layers, 3 nodes in first hidden layer, 2 nodes in second hidden layer, 3 nodes in third hidden layer,
                                                        # and finally 2 outputs
my_network


# # 2. Summation
# 
# Now that we have all the randomly assigned weights at each node for every hidden layer, we can compute the summation to feed into an activation function. Let's do that!

# In[ ]:


def summation(inputs, weights, bias):
    return np.sum(inputs * weights) + bias


# Time to define *inputs* to *my_network*

# In[ ]:


from random import seed
np.random.seed(5) #setting the seed to reproduce the same results whenever we run the code

inputs = np.around(np.random.uniform(size=4),decimals=2)

print('The inputs to my_network are {}'.format(inputs))


# In[ ]:


node_weights = my_network['layer_1']['node_1']['weights']
node_bias = my_network['layer_1']['node_1']['bias']

weighted_sum = summation(inputs, node_weights, node_bias)
print('The weighted sum at the first node in the hidden layer is {}'.format(np.around(weighted_sum[0], decimals=4)))


# # 3. Activation Function
# 
# Let's use sigmoid function to activate nodes in the layers. This activation function yields output which is just non-linear transformation of the summation.

# In[ ]:


def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))


# In[ ]:


node_output  = node_activation(summation(inputs, node_weights, node_bias))
print('The output of the first node in the hidden layer is {}'.format(np.around(node_output[0], decimals=4)))


# # 4. Forward Propagation
# 
# Let's create a function that applies the summation and node_activation functions to each node in the network and runs the data all the way to the output layer and outputs a prediction for each node in the output layer. Follow these steps in creating this forward propagation function.
# 
# 1. Start with the input layer as the input to the first hidden layer.
# 2. Compute the weighted sum at the nodes of the current layer.
# 3. Compute the output of the nodes of the current layer.
# 4. Set the output of the current layer to be the input to the next layer.
# 5. Move to the next layer in the network.
# 6. Repeat steps 2 - 4 until we compute the output of the output layer.
# 
# Forward Propagation by definition means the inputs go from the first input layer all the way to the output layer without any loops.

# In[ ]:


def forward_propagate(network, inputs):
    
    layer_inputs = list(inputs) # 1. Start with the input layer as the input to the first hidden layer
    
    for layer in network:
        
        layer_data = network[layer]
        
        layer_outputs = [] 
        for layer_node in layer_data:
        
            node_data = layer_data[layer_node]
        
            # 2. & 3. Compute the weighted sum and the output of each node at the same time 
            node_output = node_activation(summation(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))
            
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
    
        layer_inputs = layer_outputs # 4. set the output of this layer to be the input to next layer

    network_predictions = layer_outputs # 5. Move to the next layer in the network until you get an output from output layer
    return network_predictions


# # 5. Predictions
# 
# Then, we compute the network predictions.
# 
# 1. Initialize the neural network and define it's weights and biases.
# 2. Take random inputs for the given input size.
# 3. Use the forward_propagate function to run the data all the way to output layer from input layer.
# 4. Print predictions.

# In[ ]:


final_network = initialize_neural_network(4, 3, [3, 2, 3], 4)
inputs = np.around(np.random.uniform(size=4), decimals=2)
print('The input values for the neural network are {}'.format(inputs))
predictions = forward_propagate(final_network, inputs)
print('The predicted values by the network for the given input are {}'.format(predictions))


# In[ ]:




