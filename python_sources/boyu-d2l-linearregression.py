#!/usr/bin/env python
# coding: utf-8

# # Linear Regression
# contains:
# 1. basic elements of linear regression
# 2. implementation from zero
# 3. concise implementation in pytorch

# ## Basic Elements of Linear Regression
# 
# ### Model
# To keep things simple, we will start with running example in which we consider the problem of estimating the price of a house (e.g. in dollars) based on area (e.g. in square feet) and age (e.g. in years). More formally, the assumption of linearity suggests that our model can be expressed in the following form:
# 
# $$\mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b$$
# 
# 
# ### Data Set
# The first thing that we need is training data. Sticking with our running example, we'll need some collection of examples for which we know both the actual selling price of each house as well as their corresponding area and age. Our goal is to identify model parameters that minimize the error between the predicted price and the real price.   
# In the terminology of machine learning, the data set is called a training data or training set, a house (often a house and its price) here comprises one sample, and its actual selling price is called a label. The two factors used to predict the label are called features or covariates.
# 
# ### Loss Function
# In model training, we need to measure the error between the predicted value and the real value of the price. Usually, we will choose a non-negative number as the error. The smaller the value, the smaller the error. A common choice is the square function. For given parameters $\mathbf{w}$ and $b$, we can express the error of our prediction on a given a sample as follows:
# 
# $$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2,$$
# 
# $$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$
# 
# ### Optimization - Gradient Descent
# Linear regression happens to be an unusually simple optimization problem. Unlike nearly every other model that we will encounter in this book, linear regression can be solved easily with a simple formula, yielding a global optimum. 
# Even in cases where we cannot solve the models analytically, and even when the loss surfaces are high-dimensional and nonconvex, it turns out that we can still train models effectively in practice. 
# The key technique for optimizing nearly any deep learning model, and which we will call upon throughout this book, consists of iteratively reducing the error by updating the parameters in the direction that incrementally lowers the loss function.   
# $$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b)$$  
# To summarize, steps of the algorithm are the following: (i) we initialize the values of the model parameters, typically at random; (ii) we iterate over the data many times, updating the parameters in each by moving the parameters in the direction of the negative gradient, as calculated on a random minibatch of data.

# ## Vectorization for Speed
# When training our models, we typically want to process whole minibatches of examples simultaneously. Doing this efficiently requires that we vectorize the calculations and leverage fast linear algebra libraries rather than writing costly for-loops in Python.
# 
# To illustrate why this matters so much, we can consider two methods for adding vectors.
# 1. loop over vectors with Python for loop
# 2. rely on a single call to torch

# In[ ]:


import torch
import time

n = 1000
a = torch.ones(n)
b = torch.ones(n)


# In[ ]:


# define the Timer class to record time 
class Timer(object):
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        # Start the timer
        self.start_time = time.time()

    def stop(self):
        # Stop the timer and record the time in a list
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def avg(self):
        # Return the average time
        return sum(self.times)/len(self.times)

    def sum(self):
        # Return the sum of time
        return sum(self.times)


# Now we can benchmark the workloads. First, we add them, one coordinate at a time, using a for loop.

# In[ ]:


timer = Timer()
c = torch.zeros(n)
for i in range(n):
    c[i] = a[i] + b[i]
'%.5f sec' % timer.stop()


# Alternatively, we rely on torch to compute the elementwise sum:
# 

# In[ ]:


timer.start()
d = a + b
'%.5f sec' % timer.stop()


# You probably noticed that the second method is dramatically faster than the first. Vectorizing code often yields order-of-magnitude speedups. 

# ## Implementation from zero
# 
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

print(torch.__version__)


# ### Generating the Dataset

# In[ ]:



num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                      dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)


# In[ ]:


def use_svg_display():
    # display in vector graph
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # set the size of figure
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);


# ### Reading the Dataset

# In[ ]:


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # random read 10 samples
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # the last time may be not enough for a whole batch
        yield  features.index_select(0, j), labels.index_select(0, j)


# In[ ]:


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


# ### Initializing Model Parameters

# In[ ]:


w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# ### Defining the Model

# In[ ]:


def linreg(X, w, b):
    return torch.mm(X, w) + b


# ### Defining the Loss Function

# In[ ]:


def squared_loss(y_hat, y): 
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# ### Defining the Optimization Alogrithm

# In[ ]:


def sgd(params, lr, batch_size): 
    for param in params:
        param.data -= lr * param.grad / batch_size # ues .data to operate param without gradient track


# ### Training

# In[ ]:


lr = 0.03
num_epochs = 5
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # training repeats num_epochs times
    # in each epoch, all the samples in dataset will be used once
    # X is the feature and y is the label of a batch sample
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l is the loss of the batch sample
        l.backward()  # calculate the gradient of batch sample loss 
        sgd([w, b], lr, batch_size)  # using small batch random gradient descent to iter model parameters
        
        # reset parameter gradient
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))


# In[ ]:


w, true_w, b, true_b


# ## Concise Implementation in PyTorch
# 

# In[ ]:


import torch
from torch import nn
import numpy as np
torch.manual_seed(1)

print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')


# ### Generating the Dataset

# In[ ]:


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)


# ### Reading the Dataset

# In[ ]:


import torch.utils.data as Data

batch_size = 10

# combine featues and labels of dataset
dataset = Data.TensorDataset(features, labels)

# put dataset into DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,            # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # whether shuffle the data or not
    num_workers=2,              # read data in multithreading
)


# In[ ]:


for X, y in data_iter:
    print(X, '\n', y)
    break


# ### Defining the Model

# In[ ]:


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()      # call father function to init 
        self.linear = nn.Linear(n_feature, 1)  # function prototype: `torch.nn.Linear(in_features, out_features, bias=True)`

    def forward(self, x):
        y = self.linear(x)
        return y
    
net = LinearNet(num_inputs)
print(net)


# In[ ]:



# method one
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # other layers can be added here
    )

# method two
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# method three
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])


# ### Initializing the Model Parameters

# In[ ]:


from torch.nn import init

init.normal_(net[0].weight, mean=0.0, std=0.01)
init.constant_(net[0].bias, val=0.0)  # or you can use `net[0].bias.data.fill_(0)` to modify it directly


# In[ ]:


for param in net.parameters():
    print(param)


# ### Defining the Loss Function

# In[ ]:


loss = nn.MSELoss()    # nn built-in squared loss function
                       # function prototype: `torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')`


# ### Defining the Optimization Alogrithm

# In[ ]:


import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)   # built-in random gradient descent function
print(optimizer)  # function prototype: `torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)`


# ### Training

# In[ ]:


num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # reset gradient, equal to net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))


# In[ ]:


dense = net[0]
print(true_w, dense.weight.data)
print(true_b, dense.bias.data)


# In[ ]:




