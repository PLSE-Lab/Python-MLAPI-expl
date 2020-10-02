#!/usr/bin/env python
# coding: utf-8

# MNIST problem is like a Hello World of deep learning at this point, and probably everyone who is interested in neural nets goes through it eventually. So, here is my take on it. And I will be using PyTorch to solve it. I haven't yet found a detailed solution of the MNIST with PyTorch on Kaggle, so I figured I'd make my own one.
# 
# I will be using fully connected neural net and batched learning algorithm. This model is *far from perfect*, but I tried to make it as *simple* as it gets and explain every step of this solution. If you are a beginner and interested in PyTorch, hopefully, this will be helpful to you.
# 
# So, with that being said, let's start with imports that we will need.
# First of all, we need to import PyTorch. There are some common names for torch modules (like numpy is always named np):  torch.nn.functional is imported as F,  torch.nn is the core module, and is simply imported as nn.
# Also, we will need numpy, of course. Plus, I used pyplot and seaborn for some neat visualization, but they are not required for the network itself. 
# And finally, I used pandas for importing and transforming data. You can probably do it without pandas in a much better way, but I just felt like using it, so why not.

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Now we can import and transform the data. I decided to split it into input and labels right away at this step:

# In[ ]:


print("Reading the data...")
data = pd.read_csv('../input/train.csv', sep=",")
test_data = pd.read_csv('../input/test.csv', sep=",")

print("Reshaping the data...")
dataFinal = data.drop('label', axis=1)
labels = data['label']

dataNp = dataFinal.as_matrix()
labelsNp = labels.as_matrix()
test_dataNp = test_data.as_matrix()

print("Data is ready")


# Now that data is ready, we can take a look at what we're dealing with. I will be using heatmaps from seaborn, which is an excellent tool for matrix visualization. But first, since the images in the MNIST dataset are represented as a long 1d arrays of pixels, we will need to reshape it into 2d array. That's where .reshape() from numpy comes in handy. The pictures are 28 x 28 pixels, so these will be the parameters.
# 
# Let's select a couple random samples and visualize them. I will also print their labels, so we can compare images with their actual value:

# In[ ]:


plt.figure(figsize=(14, 12))

pixels = dataNp[10].reshape(28, 28)
plt.subplot(321)
sns.heatmap(data=pixels)

pixels = dataNp[11].reshape(28, 28)
plt.subplot(322)
sns.heatmap(data=pixels)

pixels = dataNp[20].reshape(28, 28)
plt.subplot(323)
sns.heatmap(data=pixels)

pixels = dataNp[32].reshape(28, 28)
plt.subplot(324)
sns.heatmap(data=pixels)

pixels = dataNp[40].reshape(28, 28)
plt.subplot(325)
sns.heatmap(data=pixels)

pixels = dataNp[52].reshape(28, 28)
plt.subplot(326)
sns.heatmap(data=pixels)

print(labels[10], " / ", labels[11])
print(labels[20], " / ", labels[32])
print(labels[40], " / ", labels[52])


# PyTorch has it's own way to store data - those are called tensors, and they are just like numpy arrays, but are suited for PyTorch needs. If we want to feed the data to the network, we need to transform the dataset into those tensors. The good news is that PyTorch can easily do that by transforming numpy arrays or regular lists into tensors.

# In[ ]:


x = torch.FloatTensor(dataNp.tolist())
y = torch.LongTensor(labelsNp.tolist())


# Before we start writing the actual network, we need to determine what will be the hyperparameters. Those will not be adjusted during training, so we need to be careful how we set them up. 
# 
# Here's what we will specify:
# * **input_size** - size of the input layer, it is always fixed (784 pixels)
# * **output_size** - size of the output layer, also fixed size (10 for every possible digit)
# * **hidden_size** - size of the hidden layer, this parameter determines structure of the network. 200 worked for me, but it is worth to play with this parameter to see what works for you
# * **epochs** - how many times will the network go through the entire dataset during training. 
# * **learning_rate** - determines how fast will the network learn. You should be very careful about this parameter, because if it is too high, the network won't learn at all, if it is too low, the net will learn too long. I's always about balance. Usualy 10^-3 - 10^-5 works just fine.
# * **batch_size** - size of mini batches during training

# In[ ]:


# hyperparameters
input_size = 784
output_size = 10
hidden_size = 200

epochs = 20
batch_size = 50
learning_rate = 0.00005


# Now we can finally write the actual network. To make it all work, the Network class needs to inherit the *nn.Module*, which gives it the basic functionality required, and allows PyTorch to work with it as expected. 
# 
# When writing a PyTorch neural network, some things must always be there:
# * \__init\__(self) - initializes the net and creates an instance of that *nn.Module*. Here we define the structure of the network.
# * forward(self, x) - defines forward propagation and how the data flow through the network. Of course, it is based on the structure that is defined in the previous function.
# 
# In the initialization, first of all, we need to initialize super (or base) module that the net inherits. After that first line, is the definition of structure. You can experiment with (put more layers or change hidden layer size, etc.), but this structure worked for me just fine.
# 
# In forward propagation we simply reassign the value of x as it flows through the layers and return the [softmax](https://en.wikipedia.org/wiki/Softmax_function) at the end.

# In[ ]:


class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l3(x)
        return F.log_softmax(x)


# After we've defined the network, we can initialize it. 
# Also, if we "print" the instance of the net, we can see the structure of it in a neat format:

# In[ ]:


net = Network()
print(net)


# Now it's time to set up the [optimizer](http://pytorch.org/docs/master/optim.html) and a loss function. 
# 
# *There are quite a lot of things happening behind these two lines of code, so if you don't know what is going on here, don't worry too much for now, it will get clearer eventualy.* 
# 
# Optimizer is what  updates the parameters of the network. I will be using Stochastic Gradient Descent with momentum. Also, the optimizer takes the network parameters as an argument, but it's not a big deal since we can get those with .parameters() function.
# 
# I decided to use [Cross Entropy Loss](https://en.wikipedia.org/wiki/Cross_entropy) for this problem, but again, there are many options and you are free to choose whatever suits you best.

# In[ ]:


optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
loss_func = nn.CrossEntropyLoss()


# Now that everything is ready, our network can start learning. I will separate data into minibatches and feed it to the network. It has many advantages over single batch learning, but that is a different story. 
# 
# Also, I will use loss_log list to keep track of the loss function during the training process. 

# In[ ]:


loss_log = []

for e in range(epochs):
    for i in range(0, x.shape[0], batch_size):
        x_mini = x[i:i + batch_size] 
        y_mini = y[i:i + batch_size] 
        
        x_var = Variable(x_mini)
        y_var = Variable(y_mini)
        
        optimizer.zero_grad()
        net_out = net(x_var)
        
        loss = loss_func(net_out, y_var)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            loss_log.append(loss.data[0])
        
    print('Epoch: {} - Loss: {:.6f}'.format(e, loss.data[0]))


# So, let's go line by line and see what is happening here:
# 
# This is the main loop that goes through all the epochs of training. An epoch is one full training on the full dataset.
# 
#     for e in range(epochs):
# This is the inner loop that simply goes through the dataset batch by batch:
# 
#     for i in range(0, x.shape[0], batch_size):
# Here is where we get the batches out of our data and simply assign them to variables for further work:
# 
#     x_mini = x[i:i + batch_size] 
#     y_mini = y[i:i + batch_size] 
#    These two lines are quite *important*. Remember I told you about tensors and how PyTorch stores data in them? That's not the end of story. Actually, to allow the network to work with data, we need a wrapper for those tensors called Variable. It has some additional properties, like allowing automatic gradient computation when backpropagating. It is required for the proper work of PyTorch, so we will add them here and supply tensors as parameters:
#    
#     x_var = Variable(x_mini)
#     y_var = Variable(y_mini)
# This line just resets the gradient of the optimizer:
#     
#     optimizer.zero_grad()
# Remember the *forward(self, x)* function that we previously defined? The next line is basically calling this function and does the forward propagation:
# 
#     net_out = net(x_var)
# This line computes the loss function based on predictions of the net and the correct answers:
# 
#     loss = loss_func(net_out, y_var)
# Here we compute the gradient based on the loss that we've got. It will be used to adjust parameters of the network.
# 
#     loss.backward()
# And here is where we finally update our network with new adjusted parameters:
# 
#     optimizer.step()
# The rest is just logging, which might be helpful to observe how well the network is performing.
# 
# After the network is done with training, we can take a look at the loss function, and how it behaved during training:

# In[ ]:


plt.figure(figsize=(10,8))
plt.plot(loss_log)


# At this point, the network should be trained, and we can make a prediction using the test dataset. All we need to do is wrap the data into the Variable and feed it to the trained net, so nothing new here.

# In[ ]:


test = torch.FloatTensor(test_dataNp.tolist())
test_var = Variable(test)

net_out = net(test_var)

print(torch.max(net_out.data, 1)[1].numpy())


# Now we have out predictions that are ready to be submitted. Before that, we can take a look at predictions and compare them to the actual pictures of digits, just like at the start with training data:

# In[ ]:


plt.figure(figsize=(14, 12))

pixels = test_dataNp[1].reshape(28, 28)
plt.subplot(321)
sns.heatmap(data=pixels)
test_sample = torch.FloatTensor(test_dataNp[1].tolist())
test_var_sample = Variable(test_sample)
net_out_sample = net(test_var_sample)

pixels = test_dataNp[10].reshape(28, 28)
plt.subplot(322)
sns.heatmap(data=pixels)
test_sample = torch.FloatTensor(test_dataNp[10].tolist())
test_var_sample = Variable(test_sample)
net_out_sample = net(test_var_sample)

pixels = test_dataNp[20].reshape(28, 28)
plt.subplot(323)
sns.heatmap(data=pixels)
test_sample = torch.FloatTensor(test_dataNp[20].tolist())
test_var_sample = Variable(test_sample)
net_out_sample = net(test_var_sample)

pixels = test_dataNp[30].reshape(28, 28)
plt.subplot(324)
sns.heatmap(data=pixels)
test_sample = torch.FloatTensor(test_dataNp[30].tolist())
test_var_sample = Variable(test_sample)
net_out_sample = net(test_var_sample)

pixels = test_dataNp[100].reshape(28, 28)
plt.subplot(325)
sns.heatmap(data=pixels)
test_sample = torch.FloatTensor(test_dataNp[100].tolist())
test_var_sample = Variable(test_sample)
net_out_sample = net(test_var_sample)

pixels = test_dataNp[2000].reshape(28, 28)
plt.subplot(326)
sns.heatmap(data=pixels)
test_sample = torch.FloatTensor(test_dataNp[1].tolist())
test_var_sample = Variable(test_sample)
net_out_sample = net(test_var_sample)


print("Prediction: {} / {}".format(torch.max(net_out.data, 1)[1].numpy()[1], torch.max(net_out.data, 1)[1].numpy()[10]))
print("Prediction: {} / {}".format(torch.max(net_out.data, 1)[1].numpy()[20], torch.max(net_out.data, 1)[1].numpy()[30]))
print("Prediction: {} / {}".format(torch.max(net_out.data, 1)[1].numpy()[100], torch.max(net_out.data, 1)[1].numpy()[2000]))


# In[ ]:


output = (torch.max(net_out.data, 1)[1]).numpy()
#np.savetxt("out.csv", np.dstack((np.arange(1, output.size+1),output))[0],"%d,%d",header="ImageId,Label")


# And that is about it, we've made a simple neural network using PyTorch that can recognize handwritten digits. Not so bad!
# 
# When I was writing this notebook, this model scorred 96.6%, which is not perfect by any means, but it's not that bad either. 
# 
# I hope this was useful for some of you. If you are totally new to deep learning, I suggest you learn how the neural networks actually work from the inside, especially the backpropagation algorithm.
# 
# These videos explain [neural nets](https://www.youtube.com/watch?v=aircAruvnKk&t=708s) and [backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U) quite well.
# 
# Also I suggest you to take a look at this [online book](http://neuralnetworksanddeeplearning.com/chap1.html) (it's absolutely free, btw), where neural networks are explained in great detail, and it even has an implementation of the MNIST problem from scratch, using only numpy.
# 
# If you have any feedback, feel free to leave comments down below, and good luck with your deep learning adventures :)
