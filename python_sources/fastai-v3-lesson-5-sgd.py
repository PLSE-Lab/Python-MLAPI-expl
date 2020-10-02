#!/usr/bin/env python
# coding: utf-8

# This is [fastai](https://www.fast.ai/) v3 [lesson 5](https://course.fast.ai/videos/?lesson=5) notebook on SGD. You can find original [here](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson5-sgd-mnist.ipynb).
# 
# I added my own (indented) comments/summary of what Jeremy Howard explains from this code in [fastai](https://www.fast.ai/) [lesson 5](https://course.fast.ai/videos/?lesson=5).

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.basics import *


# ## MNIST SGD

# Get the 'pickled' MNIST dataset from http://deeplearning.net/data/mnist/mnist.pkl.gz. We're going to treat it as a standard flat dataset with fully connected layers, rather than using a CNN.

# In[ ]:


from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)


# In[ ]:


import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")


# > We have 50000 digit images of 28x28 size which is 784 pixels, which are stored in a flat array:

# In[ ]:


x_train.shape


# > In order to show an image, we have to reshape it:

# In[ ]:


from matplotlib import pyplot as plt
import numpy as np

plt.imshow(x_train[0].reshape((28,28)), cmap="gray")


# > We are going to convert np arrays to pyTorch tensors:

# In[ ]:


import torch

x_train,y_train,x_valid,y_valid = map(torch.tensor, (x_train,y_train,x_valid,y_valid))
n,c = x_train.shape
x_train.shape, y_train.min(), y_train.max()


# In lesson2-sgd we did these things ourselves:
# 
# ```python
# x = torch.ones(n,2) 
# def mse(y_hat, y): return ((y_hat-y)**2).mean()
# y_hat = x@a
# ```
# 
# Now instead we'll use PyTorch's functions to do it for us, and also to handle mini-batches (which we didn't do last time, since our dataset was so small).

# In[ ]:


from  torch.utils.data import TensorDataset
from fastai.basic_data import DataBunch
bs=64
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
data = DataBunch.create(train_ds, valid_ds, bs=bs)


# We can iterate over data bunch training or validation set and everytime we'll get a batch of x and y.

# In[ ]:


x,y = next(iter(data.train_dl))
x.shape,y.shape


# We can define our own logistic regression model:

# In[ ]:


from torch import nn
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10, bias=True)

    def forward(self, xb): return self.lin(xb)


# In[ ]:


model = Mnist_Logistic().cuda()


# In[ ]:


model.lin


# In[ ]:


x.shape


# In[ ]:


model(x).shape


# > We look at the models parameter matrices:

# In[ ]:


[p.shape for p in model.parameters()]


# In[ ]:


lr=2e-2


# > For linear regression we use cross entropy loss. We can't use MSE because we can't compare actual results vs prediction by saying how close it is from the real value (a 4 is no more closer to a 7 than a 6).
# 
# > Cross-entropy loss is used when we have multiple classes and we are doing single label classification (e.g., dog vs cat vs horse). It requires that the predictions for each of those classes add up to one, which means that last layer must be a softmax, which ensures that property. Pytorch handles cross entropy loss and softmax together. So if you specify that your loss is cross entropy, then by default the last layer of your architecture is softmax.

# In[ ]:


loss_func = nn.CrossEntropyLoss()


# > Now given some input parameters, predictions and learning rate, we are going to update the weights of our model with weight decay.
# 
# > We calculate the loss by adding to it the sum of squared weights multiplied by the weight decay hyperparameter.
# 
# > Next we do backward propagation to calculate the gradients, then forward propagation: that is we update each parameter by substracting to it the learning rate multiplied by the parameter's gradient. We then reset parameter's gradient to zero.
# 
# > Function returns the loss.
# 
# > When we substract `lr*p.grad` from a weight, we are substracting the derivative of the loss divided by the w(t-1). That derivative of the loss is the sum of the derivatives of both arguments of the loss: so derivative of `w2.wd` can be simplified to `wd.w`. In other words we substract from the weight, `wd.w`. This is weight decay, which is another way of talking about L2 regularization (the summing up of `w2*wd` to the loss). So when we do L regularization we do weight decay.

# In[ ]:


def update(x,y,lr):
    wd = 1e-5
    y_hat = model(x)
    # weight decay
    w2 = 0.
    for p in model.parameters(): w2 += (p**2).sum()
    # add to regular loss
    loss = loss_func(y_hat, y) + w2*wd
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.sub_(lr * p.grad)
            p.grad.zero_()
    return loss.item()


# > Now we can have a loop that updates the weights given a batch of training data:

# In[ ]:


losses = [update(x,y,lr) for x,y in data.train_dl]


# In[ ]:


plt.plot(losses);


# > So with weight decay we can have a complex model but avoid overfitting, or have a small number of data and avoid overfitting.

# > We can define a two layer NN model for our Mnist data. In the forward function, we run data through first layer, than do a relu, then return output of second layer (a linear function):

# In[ ]:


import torch.nn.functional as F
class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 50, bias=True)
        self.lin2 = nn.Linear(50, 10, bias=True)

    def forward(self, xb):
        x = self.lin1(xb)
        x = F.relu(x)
        return self.lin2(x)


# In[ ]:


model = Mnist_NN().cuda()


# In[ ]:


losses = [update(x,y,lr) for x,y in data.train_dl]


# In[ ]:


len(losses)


# In[ ]:


plt.plot(losses);


# In[ ]:


model = Mnist_NN().cuda()


# > Instead of writing the details of the weight updating function, we can use a predefine optimizer like Adam: `opt.step()` is the function that does the updating.

# In[ ]:


from torch.optim import Adam,SGD
def update(x,y,lr):
    opt = SGD(model.parameters(), lr) # if using previous lr with Adam, model will diverge
    y_hat = model(x)
    loss = loss_func(y_hat, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()


# In[ ]:


losses = [update(x,y,lr) for x,y in data.train_dl]


# In[ ]:


plt.plot(losses);


# > If we use same learning rate as before and Adam optimizer, model is not converging as loss is oscillating up and down. So we need to use a different learning rate. We can try with 1e-3.
# 
# > If we use SGD optimizer, we can use same LR as before.
# 
# > When using SGD, model converges after about 800 iterations. When using Adam, model converges quicker, after 200-300 iterations.

# > We can also wrap model, data and loss in a learner. The learner does the update with default optimizer for you. With the learner you can also find the best LR.

# In[ ]:


learn = Learner(data, Mnist_NN(), loss_func=loss_func, metrics=accuracy)


# In[ ]:


get_ipython().run_line_magic('debug', '')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(1, 1e-2)


# > What `fit_one_cycle` does it vary the learning rate and momentum for every batch (?) of the cycle. The plots below show how it works. The left plot shows how LR varies with iterations. It starts of small, goes up and then back down again. The reason is when starting, we might be in a bumpy area of the curve so it's best to have a small LR, otherwise we would shoot up in all directions. As we progress and get into more stable areas of the curve, LR can be raised, before getting low again as we get closer to the convergence point. 
# 
# > The plot on the right shows how momentum varies with iterations, and exhibits the opposite of LR. That is, to start with, momentum is high, when LR is low, this is because we make little step and so we might as well make them faster by copying previous steps. As LR gets bigger, momentum becomes smaller, because we might move around a lot more and so rely more on current gradient.

# In[ ]:


learn.recorder.plot_lr(show_moms=True)


# > The `plot_losses` function below is much smoother than when we were plotting the array of losses above. This is because this fastai function uses of each loss its exponentially moving average. This makes the curve easier to interpret.

# In[ ]:


learn.recorder.plot_losses()


# ## fin
