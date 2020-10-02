#!/usr/bin/env python
# coding: utf-8

# # Forward and Backward Pass

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import torch


# ## Getting  and Processing Data 
# We will first get some data for buidling our neural network.

# In[ ]:


from pathlib import Path
import requests
import pickle
import gzip
from torch import tensor


# In[ ]:


DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)


# In[ ]:


with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
        x_train,y_train,x_valid,y_valid = map(tensor, (x_train,y_train,x_valid,y_valid))


# In[ ]:


def normalize(x, m, s): return (x-m)/s


# Calculating Mean and Standard Deviation for Normalizing data, this is very important for model to perform well.

# In[ ]:


train_mean,train_std = x_train.mean(),x_train.std()
train_mean,train_std


# Examining various parameters, input size, number of classes, etc. We have 50000 images of 784 inputs, minimum value of y is 0 and max value is 9.

# In[ ]:


n,c = x_train.shape
x_train, x_train.shape, y_train, y_train.shape, y_train.min(), y_train.max()


# In[ ]:


import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['image.cmap'] = 'gray'


# In[ ]:


img = x_train[0]
img.view(28,28).type()


# Looking at one of the supplied images.

# In[ ]:


plt.imshow(img.view((28,28)))


# ## Building the network
# 
# First model to begin with is y = wx + b and we will have a hidden layer with 50 as the number of nodes. This is defined as nh.

# In[ ]:


n,m = x_train.shape
c = y_train.max()+1
nh = 50
n,m,c, nh


# Since we have a hidden layer with 50 activations, there are weights w1 and w2 are required with dimensions m by nh for first section and nh by 1 for section 2.

# In[ ]:


w1 = torch.randn(m,nh)/math.sqrt(m)
b1 = torch.zeros(nh)
w2 = torch.randn(nh,1)/math.sqrt(nh)
b2 = torch.zeros(1)


# At one node, two things happen normally, linear transformation and activation (normally relu).
# Defining the linear transformation:

# In[ ]:


def lin(x, w, b): return x@w + b
def relu(x): return x.clamp_min(0.)


# In[ ]:


# Doing transformation here on x_valid.
t = relu(lin(x_valid, w1, b1))


# In[ ]:


def model(xb):
    l1 = lin(xb, w1, b1)
    l2 = relu(l1)
    l3 = lin(l2, w2, b2)
    return l3


# This is the model! If you see only three lines of code does that. This can also be said to have forward pass of the neural network. But essentially, building blocks are in place. One can extend the model with more layers, nh1, nh2 and so on.
# There is lot to do yet. We need to do backward pass and calculate the gradients and then update parameters such as w1, b1 and so on. 

# In[ ]:


get_ipython().run_line_magic('timeit', '-n 10 _=model(x_valid)')


# ### Loss function
# Since we are treating this as a regression problem, see the output has only one node earlier, we will create mse as our loss function.

# In[ ]:


def mse(output, targ): return (output.squeeze(-1) - targ).pow(2).mean()


# In[ ]:


# Calculating the mse for our initial model as built above,
y_train,y_valid = y_train.float(),y_valid.float()
mse(model(x_train),y_train)


# ### Backward Pass
# Perhaps the most important section of building a neural network. During the backward pass, weights are updated so that loss function is minimized. A neural network performs best when the weights are available in such manner that when we apply them to a input set and do a forward pass, then loss is at minimum. Lets implement the same.

# In[ ]:


def mse_grad(inp, targ): 
    # grad of loss with respect to output of previous layer
    inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]


# In[ ]:


# Defining gradient for relu layer:
def relu_grad(inp, out):
    # grad of relu with respect to input activations
    # this is 1 when input is greater than zero else 0.
    inp.g = (inp>0).float() * out.g


# In[ ]:


# Defining linear gradient
def lin_grad(inp, out, w, b):
    inp.g = out.g @ w.t()
    w.g = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(0)
    b.g = out.g.sum(0)


# In[ ]:


# Implementing chain rule of derivatives will help in performing the back propagation, here is the full implementation of code.


# In[ ]:


def forward_and_backward(inp, targ):
    # forward pass:
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2
    # loss calculation after forward pass.
    loss = mse(out, targ)
    
    # backward pass:
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)


# In[ ]:





# In[ ]:


# Lets run forward and backward once and see what happens.
forward_and_backward(x_train, y_train)


# In[ ]:


# We are saving to check our implementation and check with Pytorch standard implementation later on.
w1g = w1.g.clone()
w2g = w2.g.clone()
b1g = b1.g.clone()
b2g = b2.g.clone()
ig  = x_train.g.clone()


# In[ ]:


xt2 = x_train.clone().requires_grad_(True)
w12 = w1.clone().requires_grad_(True)
w22 = w2.clone().requires_grad_(True)
b12 = b1.clone().requires_grad_(True)
b22 = b2.clone().requires_grad_(True)


# In[ ]:


def forward(inp, targ):
    # forward pass:
    l1 = inp @ w12 + b12
    l2 = relu(l1)
    out = l2 @ w22 + b22
    # we don't actually need the loss in backward!
    return mse(out, targ)


# In[ ]:


loss = forward(xt2, y_train)


# In[ ]:


loss.backward()


# In[ ]:


# Lets look at some of the gradients now.
w22.grad


# In[ ]:


w2g


# Looks like we have done a good job on finding the gradients. One thing to notice that with loss.backward(), we do not need to calculate the gradients by writing code. This is provided by pytorch.

# That's it for now. Thanks for reading. Please do send your feedback using comments! Thank you!
# 
# Note: This Kernel is produced by following Jeremy Howard's FastAI course. You can check this out at fast.ai
# 
