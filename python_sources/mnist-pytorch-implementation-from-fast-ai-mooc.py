#!/usr/bin/env python
# coding: utf-8

# ## Using SGD on MNIST
# 
# Here I am just implementing the notebook from fast.ai's MOOC on ML : http://course18.fast.ai/lessonsml1/lesson8.html.
# I also wanted to try out the GPUs on Kaggle kernels so this seemed like the ideal notebook to replicate and experiment with.

# <img src="images/mnist.png" alt="" style="width: 60%"/>

# ## Imports and data

# In[ ]:


get_ipython().system('pip3 install ipywidgets')
get_ipython().system('jupyter nbextension enable --py --sys-prefix widgetsnbextension')


# In[ ]:


#!pip install --upgrade pip
#!pip install fastai==0.7.0    ## Installed from personal Github repo to avoid numpy rounding error : 
                               ## https://forums.fast.ai/t/unfamiliar-error-when-running-learn-fit/35075/19
get_ipython().system('pip install torchtext==0.2.3')
#!pip intall numpy==1.15.1   ## attirbute error thrown due to numpy updates. Changed fastai source code though
get_ipython().system('pip install Pillow==4.1.1')
get_ipython().system('pip install blosc')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.imports import *
from fastai.torch_imports import *
from fastai.io import *


# Let's download, unzip, and format the data.

# In[ ]:


import os
import pandas as pd
import pickle
import gzip


# In[ ]:


((x, y), (x_valid, y_valid), _) = pickle.load(gzip.open('../input/mnist.pkl.gz', 'rb'), encoding='latin-1')


# In[ ]:


type(x), x.shape , type(y), y.shape


# ### Normalize

# Many machine learning algorithms behave better when the data is *normalized*, that is when the mean is 0 and the standard deviation is 1. We will subtract off the mean and standard deviation from our training set in order to normalize the data:

# In[ ]:


mean = x.mean()
std = x.std()

x=(x-mean)/std
mean, std, x.mean(), x.std()


# Note that for consistency (with the parameters we learn when training), we subtract the mean and standard deviation of our training set from our validation set. 

# In[ ]:


x_valid = (x_valid-mean)/std
x_valid.mean(), x_valid.std()


# ### Look at the data

# In any sort of data science work, it's important to look at your data, to make sure you understand the format, how it's stored, what type of values it holds, etc. To make it easier to work with, let's reshape it into 2d images from the flattened 1d format.

# #### Helper methods

# In[ ]:


def show(img, title=None):
    plt.imshow(img, cmap="gray")
    if title is not None: plt.title(title)


# In[ ]:


def plots(ims, figsize=(12,6), rows=2, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], cmap='gray')


# #### Plots 

# In[ ]:


x_valid.shape


# In[ ]:


x_imgs = np.reshape(x_valid, (-1,28,28)); x_imgs.shape


# In[ ]:


show(x_imgs[0], y_valid[0])


# In[ ]:


y_valid.shape


# It's the digit 3!  And that's stored in the y value:

# In[ ]:


y_valid[0]


# We can look at part of an image:

# In[ ]:


x_imgs[0,10:15,10:15]


# In[ ]:


show(x_imgs[0,10:15,10:15])


# In[ ]:


plots(x_imgs[:8], titles=y_valid[:8])


# ## Neural Net in PyTorch

# In[ ]:


from fastai.metrics import *
from fastai.model import *
from fastai.dataset import *

import torch.nn as nn


# We will begin with the highest level abstraction: using a neural net defined by PyTorch's Sequential class.  

# In[ ]:


net = nn.Sequential(
    nn.Linear(28*28, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.LogSoftmax()
#).cuda()  ## For GPU
)         ## For CPU


# Each input is a vector of size `28*28` pixels and our output is of size `10` (since there are 10 digits: 0, 1, ..., 9). 
# 
# We use the output of the final layer to generate our predictions.  Often for classification problems (like MNIST digit classification), the final layer has the same number of outputs as there are classes.  In that case, this is 10: one for each digit from 0 to 9.  These can be converted to comparative probabilities.  For instance, it may be determined that a particular hand-written image is 80% likely to be a 4, 18% likely to be a 9, and 2% likely to be a 3.

# In[ ]:


md = ImageClassifierData.from_arrays('../input/mnist.pkl.gz', (x,y), (x_valid, y_valid))


# In[ ]:


loss=nn.NLLLoss()
metrics=[accuracy]
# opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9)
opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9, weight_decay=1e-3)


# ### Loss functions and metrics

# In[ ]:


def binary_loss(y, p):
    return np.mean(-(y * np.log(p) + (1-y)*np.log(1-p)))


# In[ ]:


acts = np.array([1, 0, 0, 1])
preds = np.array([0.9, 0.1, 0.2, 0.8])
binary_loss(acts, preds)


# Why not just maximize accuracy? The binary classification loss is an easier function to optimize.
# 
# For multi-class classification, we use *negative log liklihood* (also known as *categorical cross entropy*) which is exactly the same thing, but summed up over all classes.

# ### Fitting the model

# *Fitting* is the process by which the neural net learns the best parameters for the dataset.

# In[ ]:


fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)


# In[ ]:


set_lrs(opt, 1e-2)


# In[ ]:


fit(net, md, n_epochs=3, crit=loss, opt=opt, metrics=metrics)


# In[ ]:


fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)


# In[ ]:


set_lrs(opt, 1e-2)


# In[ ]:


fit(net, md, n_epochs=3, crit=loss, opt=opt, metrics=metrics)


# In[ ]:


t = [o.numel() for o in net.parameters()]
t, sum(t)


# In[ ]:


preds = predict(net, md.val_dl)


# In[ ]:


preds.shape


# In[ ]:


preds.argmax(axis=1)[:5]


# In[ ]:


preds = preds.argmax(1)


# Let's check how accurate this approach is on our validation set. 

# In[ ]:


np.mean(preds == y_valid)


# Let's see how some of our predictions look!

# In[ ]:


plots(x_imgs[:8], titles=preds[:8])

