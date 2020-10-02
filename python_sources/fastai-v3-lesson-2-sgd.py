#!/usr/bin/env python
# coding: utf-8

# [Lesson Video Link](https://course.fast.ai/videos/?lesson=2)
# 
# [Lesson resources and updates](https://forums.fast.ai/t/lesson-2-official-resources-and-updates/28630)
# 
# [Lesson chat](https://forums.fast.ai/t/lesson-2-chat/28722)
# 
# [Further discussion thread](https://forums.fast.ai/t/lesson-2-further-discussion/28706)
# 
# Note: This is a mirror of the FastAI Lesson 2 Nb. 
# Please thank the amazing team behind fast.ai for creating these, I've merely created a mirror of the same here
# For complete info on the course, visit course.fast.ai

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.basics import *


# In this part of the lecture we explain Stochastic Gradient Descent (SGD) which is an **optimization** method commonly used in neural networks. We will illustrate the concepts with concrete examples.

# ## Linear Regression problem
# The goal of linear regression is to fit a line to a set of points.

# In[ ]:


n=100


# In[ ]:


x = torch.ones(n,2) 
x[:,0].uniform_(-1.,1)
x[:5]


# In[ ]:


a = tensor(3.,2); a


# In[ ]:


y = x@a + torch.rand(n)


# In[ ]:


plt.scatter(x[:,0], y);


# You want to find **parameters** (weights) a such that you minimize the error between the points and the line x@a. Note that here a is unknown. For a regression problem the most common error function or loss function is the **mean squared error**.

# In[ ]:


def mse(y_hat, y): return ((y_hat-y)**2).mean()


# Suppose we believe a = (-1.0,1.0) then we can compute y_hat which is our prediction and then compute our error.

# In[ ]:


a = tensor(-1.,1)


# In[ ]:


y_hat = x@a
mse(y_hat, y)


# In[ ]:


plt.scatter(x[:,0],y)
plt.scatter(x[:,0],y_hat);


# 
# So far we have specified the *model* (linear regression) and the *evaluation criteria* (or loss function). Now we need to handle *optimization*; that is, how do we find the best values for a? How do we find the best *fitting* linear regression.
# 

# ## Gradient Descent
# 
# We would like to find the values of a that minimize mse_loss.
# 
# **Gradient descent** is an algorithm that minimizes functions. Given a function defined by a set of parameters, gradient descent starts with an initial set of parameter values and iteratively moves toward a set of parameter values that minimize the function. This iterative minimization is achieved by taking steps in the negative direction of the function gradient.
# 
# Here is gradient descent implemented in [PyTorch](https://pytorch.org).

# In[ ]:


a = nn.Parameter(a); a


# In[ ]:


def update():
    y_hat = x@a
    loss = mse(y, y_hat)
    if t % 10 == 0: print(loss)
    loss.backward()
    with torch.no_grad():
        a.sub_(lr * a.grad)
        a.grad.zero_()


# In[ ]:


lr = 1e-1
for t in range(100): update()


# In[ ]:


plt.scatter(x[:,0],y)
plt.scatter(x[:,0],x@a);


# ## **Animate it!**

# In[ ]:


from matplotlib import animation, rc
rc('animation', html='jshtml')


# In[ ]:


a = nn.Parameter(tensor(-1.,1))

fig = plt.figure()
plt.scatter(x[:,0], y, c='orange')
line, = plt.plot(x[:,0], x@a)
plt.close()

def animate(i):
    update()
    line.set_ydata(x@a)
    return line,

animation.FuncAnimation(fig, animate, np.arange(0, 100), interval=20)


# In practice, we don't calculate on the whole file at once, but we use mini-batches.
# 
# ## Vocab
# 
#  -  Learning rate
#  -  Epoch
#  -  Minibatch
#  -  SGD
#  -  Model / Architecture
#  -  Parameters
#  -  Loss function
# 
# For classification problems, we use *cross entropy loss*, also known as *negative log likelihood loss*. This penalizes incorrect confident predictions, and correct unconfident predictions.

# In[ ]:




