#!/usr/bin/env python
# coding: utf-8

# ![](https://www.doc.ic.ac.uk/~js4416/163/website/img/neural-networks/nn-model.png)

# Creating the NN architecture therefore means coming up with values for the number of layers of each type and the number of nodes in each of these layers.
# 
# **The Input Layer**
# 
# Simple--every NN has exactly one of them--no exceptions that I'm aware of.
# 
# With respect to the number of neurons comprising this layer, this parameter is completely and uniquely determined once you know the shape of your training data. Specifically, the number of neurons comprising that layer is equal to the number of features (columns) in your data. Some NN configurations add one additional node for a bias term.
# 
# **The Output Layer**
# 
# Like the Input layer, every NN has exactly one output layer. Determining its size (number of neurons) is simple; it is completely determined by the chosen model configuration.
# 
# If the NN is a regressor, then the output layer has a single node.
# 
# If the NN is a classifier, then it also has a single node unless softmax is used in which case the output layer has one node per class label in your model.
# 
# **The Hidden Layers**
# 
# There are really two decisions that must be made regarding the hidden layers: how many hidden layers to actually have in the neural network and how many neurons will be in each of these layers. We will first examine how to determine the number of hidden layers to use with the neural network.
# 
# Problems that require two hidden layers are rarely encountered. However, neural networks with two hidden layers can represent functions with any kind of shape. There is currently no theoretical reason to use neural networks with any more than two hidden layers. In fact, for many practical problems, there is no reason to use any more than one hidden layer.
# 

# ![Hidden Layer](https://image.ibb.co/ffqgUH/hidden_layer.png)

# In this post, I would like to ask what is the minimum number of neurons and layers that are needed for the classification of simple separable features. While this is not necessarily a new problem, I will explore a few interesting aspects of this problem using Keras.
# 
# In case you wondered, and in general, the answer is that 1-layer network can represent half-planes (i.e., the AND or OR logic), 2-layer network can classify points inside any number of arbitrary lines, and 3-layer network can classify any arbitrary shape in arbitrary dimensions. Hence, a 2-layer model can classify any convex set, while 3-layer model can classify any number of disjoint convex or concave shapes.

# In[ ]:


#loading Keras, numpy, and matplotlib.
from keras import models
from keras import layers

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3107)


# In[ ]:


# constants
npts = 100 # points per blob
tot_npts = 4*npts # total number of points 
s = 0.005 # ~standard deviation
sigma = np.array([[s, 0], [0, s]]) #cov matrix


# **The AND problem**

# The AND problem is simple. As you can see below, the data is clustered around four areas: [0,0], [0,1], [1,0], and [1,1]. When we apply the AND logic function to each pair, it follows that [0,0]=0, [0,1]=0, [1,0]=0, but [1,1]=1. We label the data pair as one (blue) when both points are equal to one. Otherwise, we label the data as zero (red).

# In[ ]:


# Generate Data
data1 = np.random.multivariate_normal( [0,0], sigma, npts)
data2 = np.random.multivariate_normal( [0,1], sigma, npts)
data3 = np.random.multivariate_normal( [1,0], sigma, npts)
data4 = np.random.multivariate_normal( [1,1], sigma, npts)

and_data = np.concatenate((data1, data2, data3, data4)) # data
and_labels = np.concatenate((np.ones((3*npts)),np.zeros((npts)))) # labels
#print(and_data.shape)
#print(and_labels.shape)

plt.figure(figsize=(20,10))
plt.scatter(and_data[:,0][and_labels==0], and_data[:,1][and_labels==0],c='b')
plt.scatter(and_data[:,0][and_labels==1], and_data[:,1][and_labels==1],c='r')

plt.plot()
plt.title('AND problem',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid('on')
plt.show()
#(400, 2)
#(400,)


# Separating the AND data is easy. A straight line can separate the data into the blues and reds.
# 
# That was easy.
# 
# Linear line is represented by Wx + b (where W is the slope, and b is the bias), and in the neural network world as np.dot(W, x) + b. Thus, one layer with one neuron (i.e., one linear line) will suffice for separating the AND data.
# 
# Below you can see my Keras implementation of a neural network with one layer, one neuron, and a sigmoid activation. As a loss function, I chose binary_crossentropy with Adam optimizer. Iterating over the data with batch_size of 16, the model converges to the right solution, measured by accuracy, after about 100 iterations over the whole data.

# In[ ]:


model = models.Sequential()
model.add(layers.Dense(1, activation='sigmoid', input_shape=(2,)))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(and_data, 
                    and_labels, 
                    epochs=200,
                    batch_size=16,
                    verbose=0)
history_dict = history.history
history_dict.keys()


# In[ ]:


plt.figure(figsize=(20,10))
plt.subplot(121)
loss_values = history_dict['loss']
#val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
#plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training loss',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.subplot(122)
acc_values = history_dict['acc']
#val_acc_values = history_dict['val_acc']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, acc_values, 'bo', label='Training acc')
#plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training accuracy',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# **The OR problem**

# The OR problem is also simple. Again, the data is clustered around four areas: [0,0], [0,1], [1,0], and [1,1]. As before, we apply the OR logic function to each pair. It follows that [0,0]=0, but [0,1]=1, [1,0]=1, and [1,1]=1. We label the data pair as zero (red), only when both points are equal to zero. Otherwise, we label the data as one (blue).

# In[ ]:


# Generate Data
data1 = np.random.multivariate_normal( [0,0], sigma, npts)
data2 = np.random.multivariate_normal( [0,1], sigma, npts)
data3 = np.random.multivariate_normal( [1,0], sigma, npts)
data4 = np.random.multivariate_normal( [1,1], sigma, npts)

or_data = np.concatenate((data1, data2, data3, data4))
or_labels = np.concatenate((np.ones((npts)),np.zeros((3*npts))))

plt.figure(figsize=(20,10))
plt.scatter(or_data[:,0][or_labels==0], or_data[:,1][or_labels==0],c='b')
plt.scatter(or_data[:,0][or_labels==1], or_data[:,1][or_labels==1],c='r')
plt.title('OR problem',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid('on')
plt.show()


# Separating this data is also straightforward. As for the AND data, a straight line will suffice and as before a neural network with one layer and one neuron is the minimum model we need to separate or classify the data correctly. Using the same architecture as for the AND problem, you can see that the model converges to the right solution after about 300 iterations. As a side note, let me just mention that the number of iterations is not important for this post as we just look for a model that can yield 100% accuracy.

# In[ ]:


model = models.Sequential()
model.add(layers.Dense(1, activation='sigmoid', input_shape=(2,)))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(or_data, 
                    or_labels, 
                    epochs=400,
                    batch_size=16,
                    verbose=0)


history_dict = history.history
history_dict.keys()


plt.figure(figsize=(20,10))
plt.subplot(121)
loss_values = history_dict['loss']
#val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
#plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training loss',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.subplot(122)
acc_values = history_dict['acc']
#val_acc_values = history_dict['val_acc']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, acc_values, 'bo', label='Training acc')
#plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training accuracy',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# **The XOR problem**

# The XOR problem is a bit more difficult. Again, the data points are clustered around four areas, and we apply the XOR logic function to each pair. For the XOR logic the result is that [0,0]=0, and [1,1]=0 but [0,1]=1, and [1,0]=1.

# In[ ]:


# Generate Data
data1 = np.random.multivariate_normal( [0,0], sigma, npts)
data2 = np.random.multivariate_normal( [0,1], sigma, npts)
data3 = np.random.multivariate_normal( [1,0], sigma, npts)
data4 = np.random.multivariate_normal( [1,1], sigma, npts)

xor_data = np.concatenate((data1, data4, data2, data3))
xor_labels = np.concatenate((np.ones((2*npts)),np.zeros((2*npts))))

plt.figure(figsize=(20,10))
plt.scatter(xor_data[:,0][xor_labels==0], xor_data[:,1][xor_labels==0],c='b')
plt.scatter(xor_data[:,0][xor_labels==1], xor_data[:,1][xor_labels==1],c='r')
plt.title('XOR problem',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid('on')
plt.show()


# The problem is that no one straight line can separate the data correctly. However, if as a first step we isolate [0,0] and [1,1] separately using two linear lines, then as a second step, we can apply the AND function to both separations and the overlapped area gives us the right classification. Thus, a two-step solution is needed: the first applies two linear lines, and the second unite the two separations using an AND logic. Other words, the minimal network is a two-layer neural network, where the first must have two neurons (i.e., two linear lines) and the second only one (i.e., applying the AND logic and before we showed that this requires only one neuron).

# In[ ]:


model = models.Sequential()
model.add(layers.Dense(1, activation='sigmoid', input_shape=(2,)))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xor_data, 
                    xor_labels, 
                    epochs=400,
                    batch_size=32,
                    verbose=0)
history_dict_10 = history.history

model = models.Sequential()
model.add(layers.Dense(1, activation='relu', input_shape=(2,)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xor_data, 
                    xor_labels, 
                    epochs=400,
                    batch_size=32,
                    verbose=0)
history_dict_11 = history.history

model = models.Sequential()
model.add(layers.Dense(2, activation='relu', input_shape=(2,)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xor_data, 
                    xor_labels, 
                    epochs=400,
                    batch_size=32,
                    verbose=0)
history_dict_21 = history.history


history = model.fit(xor_data, 
                    xor_labels, 
                    epochs=400,
                    batch_size=32,
                    verbose=0)
history_dict_41 = history.history


# To test for the minimal set of two layers with two and one neurons (denoted as 2_1), I also run two more Keras implementations with a fewer number of layers and neurons. Indeed you can see that the 2_1 model (two layers with two and one neurons) converges to the right solution, while the 1_0 (one layer with one neuron) and 1_1 models (two layers with one and one neurons) approximate the data quite well but never converge to 100% accuracy. So, although it is not a formal proof that covers all aspects of the minimal set that required for classification, it should give you enough hands-on intuition into why the minimal set requires only two layers with two and one neurons.

# In[ ]:


plt.figure(figsize=(20,10))
plt.subplot(121)
loss_values = history_dict_10['loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, history_dict_10['loss'], 'o', label='Training loss')
plt.plot(epochs, history_dict_11['loss'], 'o', label='Training loss')
plt.plot(epochs, history_dict_21['loss'], 'o', label='Training loss')
plt.title('Training loss',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.legend(['1_0','1_1','2_1','4_1'],fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.subplot(122)
acc_values = history_dict_10['loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, history_dict_10['acc'], 'o', label='Training loss')
plt.plot(epochs, history_dict_11['acc'], 'o', label='Training loss')
plt.plot(epochs, history_dict_21['acc'], 'o', label='Training loss')
plt.title('Training accuracy',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(['1_0','1_1','2_1','4_1'],fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# **Reference Links**

# 1. https://en.wikipedia.org/wiki/Artificial_neural_network
# 1. https://www.doc.ic.ac.uk/~js4416/163/website/neural-networks/
# 1. https://cs.stackexchange.com/questions/2049/how-many-layers-should-a-neural-network-have
# 1. https://www.researchgate.net/post/How_to_decide_the_number_of_hidden_layers_and_nodes_in_a_hidden_layer
# 1. https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# 1. https://medium.com/@naftalicohen/how-many-layers-do-you-need-31b6d8038941

# In[ ]:




