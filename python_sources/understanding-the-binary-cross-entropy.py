#!/usr/bin/env python
# coding: utf-8

# # Understanding the binary cross entropy

# I have seen that most of the public kernels in this competition are using the **binary cross-entropy** as the loss function. Thus, I think it is very important that we fully understand how this loss works.
# 
# This is a loss function designed for **binary classification** problems, where the classifier can return the predicted probabilities of the positive class. The binary cross-entropy is defined as:
# $$
# \sum_{i=1}^{N} l(y_i, p(x_i)) = \sum_{i=1}^{N} - \left( y_i \cdot \log(p(x_i)) + (1-y_i) \cdot \log(1-p(x_i))  \right)
# $$
# where $y_i$ is the target variable (can only be $0$ or $1$), $x_i$ are the variables/features, and **$p$ the probability model of being of class $1$**. 
# 
# You can see that with a sample of the class 1:
# $$
# l(y_i=1, p(x_i)) = - \log(p(x_i))
# $$
# and thus the loss is $0$ when $p(x_i) = 1$ (the sample is of class 1, and the model outputs a probability 1 of being of class 1), and the loss is $\infty$ when $p(x_i) = 0$ (the sample is of class 1, and the model outputs a probability 0 of being of class 1). 
# 
# On the other hand, when the sample is of the class 0:
# $$
# l(y_i=0, p(x_i)) = - \log(1-p(x_i))
# $$
# and thus the loss is $0$ when $p(x_i) = 0$ (the sample is of class 0, and the model outputs a probability 0 of being of class 1), and the loss is $\infty$ when $p(x_i) = 1$ (the sample is of class 0, and the model outputs a probability 1 of being of class 1).

# ***
# ### Let's see it graphically

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# I first define a function to compute $l(y_i, p(x_i))$ in a pairwise way:

# In[ ]:


def cross_entropy(yreal, ypred):
    with np.errstate(divide='ignore', invalid='ignore'):
        return -1*(yreal*np.log(ypred) + (1-yreal)*np.log(1-ypred))


# Below you can see the values of $l(y_i, p(x_i))$ when all the samples are of class 1

# In[ ]:


yreal = np.ones(50)
ypred = np.linspace(0., 1., 50)

plt.figure(figsize=(15,5))
plt.plot(ypred, cross_entropy(yreal, ypred), "o--")
plt.xlabel("prediction", fontsize=23)
plt.ylabel("loss", fontsize=23)
plt.grid()
plt.show()


# and the values of $l(y_i, p(x_i))$ when all the samples are of class 0

# In[ ]:


yreal = np.zeros(50)
ypred = np.linspace(0., 1., 50)

plt.figure(figsize=(15,5))
plt.plot(ypred, cross_entropy(yreal, ypred), "o--")
plt.xlabel("prediction", fontsize=23)
plt.ylabel("loss", fontsize=23)
plt.grid()
plt.show()


# ***
# ### What does it happen when we use binary cross entropy and $y$ is not binary? 
# 
# In this competition we are using binary cross-entropy, however, the target variable can take any value between 0 and 1. Let's see what happen with this loss function as we move the true target value between 0 and 1: 

# In[ ]:


for x in np.arange(0, 1.01, 0.1):
    yreal = x*np.ones(50)
    ypred = np.linspace(0., 1., 50)

    plt.figure(figsize=(15,5))
    plt.plot(ypred, cross_entropy(yreal, ypred), "o--")
    plt.title(f"true target: {x}", fontsize=19)
    plt.xlabel("prediction", fontsize=23)
    plt.ylabel("loss", fontsize=23)
    plt.grid()
    plt.show()


# It seems ok, right? The loss function takes its minimum when the prediction is equal to the true target. But there are two "little" problems:
# 1. **The loss is not symmetric!**. For example, when the true target is 0.4, a sample with prediction 0 will get a higher penalization than a sample with prediction 0.8, but both are equidistant to the true target.
# 2. And even more important, **the minimum value of the loss is not 0 when the true target is not 0 or 1** (see carefully the y-axis). 
# 
# Furthermore, the minimum value of the loss **depends on the true target value!**. If you do the math, you will see that $l(y_i, p(x_i))$ reaches its minimum when $p(x_i) = y_i$. 
# 
# The figure in the cell below shows the minimal loss as a function of the true target value:

# In[ ]:


true_target = np.linspace(0.001, 0.999, 50)
min_loss = [cross_entropy(x, x) for x in true_target]

plt.figure(figsize=(15,5))
plt.plot(true_target, min_loss, "o--")
plt.xlabel("True target", fontsize=23)
plt.ylabel("Minimal loss", fontsize=23)
plt.grid()
plt.show()


# For these two reasons, I think that binary cross-entropy is **not an appropriate loss function for this problem**. 
# 
# What do you think?
# ***
