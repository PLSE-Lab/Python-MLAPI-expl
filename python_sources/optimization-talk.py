#!/usr/bin/env python
# coding: utf-8

# ## Optimization Algorithms

# **Purpose of Optimization Algorithms**
# 1. Optimization algorithms help us to minimize or maximize an objective
# 2. In the ML scenario most of the times we are trying to optimize loss w.r.t to the parameters of the model
# 3. Unless the solution is directly computable (Like in case of Linear Regression), some sort of optimization algorithm is used
# 

# **Types of Optimization Algorithms**
# 1. Generally optimization algorithms work by picking a random initial point and trying to follow a direction of decreasing loss function
# 2. Two main types of optimization algorithms: First order optimization algorithms, Second order optimization algorithms
# 3. Mainly differ in the order derivates used to find the direction of descent

# In[ ]:


from IPython.display import Image
Image("../input/optimization-talk-images/gradient_descent.png")


# **First Order Optimization Algorithms**
# 1. Uses first derivative to find the direction of descent
# 2. Gradient descent algorithms, Simplex Algorithm Procedures etc
# 
# 
# \begin{equation*}
# Update Rule : w = w - \lambda*\nabla J(w)
# \end{equation*}

# **Second order optimization algorithms**
# 1. Uses second order derivative while computing the direction of descent
# 2. Faster convergence rates but very hard to compute since the Hessian needs to be computed at every update step
# 3. Uses the curvature of the loss surface to figure out the direction. The hession gives the information of how the gradient changes in different direction
# 4. Doesn't get struck in the slow convergence paths or saddle point like first order algorithms
# 5. Newton, Quasi-Newton methods, L-BFGS etc
# 

# ## Imports

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms


# In[ ]:


path = './data/'


# ## Dataset Preparation

# In[ ]:


get_ipython().system('tar -zxvf ../input/cifar10-python/cifar-10-python.tar.gz')


# In[ ]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='.', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='.', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ##  Model Definition

# In[ ]:


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv_layers = nn.Sequential(*[
            *self.conv_block(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=1),
        ])
        self.classifier = nn.Linear(2304, 10)
        
    def conv_block(self, in_channels, out_channels):
        return [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# In[ ]:


def train_model(mod, optimizer, trn_loader, nepochs=1, verbose=True):
    crit = nn.CrossEntropyLoss()
    loss_arr = []
    for epoch in range(nepochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        for i, data in enumerate(trn_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = Variable(inputs, requires_grad=False).cuda(), Variable(labels, requires_grad=False).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = mod(inputs)
            loss = crit(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 99:    # print every 2000 mini-batches
                if verbose:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                loss_arr.append((epoch*len(trn_loader) + i+1, running_loss / 100))
                running_loss = 0.0
    print('Finished Training')
    return loss_arr


# In[ ]:


def ploterr(arr):
    xs = [x[0] for x in arr]
    ys = [x[1] for x in arr]
    plt.figure(figsize=(15,10))
    plt.subplot(1, 1, 1)
    plt.plot(xs, ys, )


# ## Optimizing Algorithms

# ### GD

# \begin{equation*}
# Update Rule : w = w - \lambda*\nabla J(w)
# \end{equation*}

# **Batch Gradient Descent**
# 1. Compute gradient for whole dataset at once
# 2. Computationally intractable when data doesn't fit to memory and are generally slow since we are computing gradient for whole dataset at once
# 3. Guarenteed to converge to global minima for convex losses, to a local minima for non convex functions

# **Stochastic Gradient Descent**
# 1. In the opposite end to BGD, here update gradient for each example.
# 2. This can be viewed as estimating the gradient on whole dataset using single data point
# 3. Since we are using single point to estimate gradient, fluctuations in gradient directions w.r.t optimal directions is huge, so we end off-shooting in sub-optimal directions in between
# 4. But starting with a small lr and gradually increasing it allows SGD to perform as good as BGD
# 5. Sometimes the huge variance of SGD allows it to find better local minima than BGD

# **Mini-Batch gradient descent**
# 1. Updates parameters in batches
# 2. Tries to incorporate best of both worlds. Reduces variance compared to SGD but allows to use highly optimized matrix operations to make the updates fatser
# 3. Widely used variant for most of the Neural Network optimization

# **Challenges**
# 1. Vanilla MBGD algos doesn't guarentee good convergence in practical
# 2. Choosing a proper lr is generally difficult, too small ends up with slow convergence, too large ends up in erratic behaviour
# 3. Learning rate schedulers are generally used to anneal the lr's as training progresses. Either pre-defined scheudes or schedules depending on the objective are used
# 4. GD methods are susceptable to saddle points and local minima which makes the training harder

# In[ ]:


net = CifarNet().cuda(0)
optimizer = opt.SGD(net.parameters(), lr=0.001)
sgd_res = train_model(net, optimizer, trainloader, verbose=False)
ploterr(sgd_res)


# ### Momentum

# \begin{equation*}
# Update Equations\\
#  v_t = \gamma * v_{t-1} + \lambda * \nabla J(w) \\
#  w_t = w_{t-1} - v_t
# \end{equation*}
# 
# 1. Accumulates past gradient in the gradient calculation step as momentum.
# 2. Dampens oscillations in irrelevant directions and accelrates in the relavant directions
# 3. Works as a moving average of gradient, dampening the variance due to mini-batched gd
# 4. Helps in moving faster in the ravines which are prevalent near local-minima

# In[ ]:


Image("../input/optimization-talk-images/momentum.png")


# In[ ]:


Image("../input/optimization-talk-images/ravines.gif")


# In[ ]:


Image("../input/optimization-talk-images/with_momentum.gif")


# In[ ]:


net = CifarNet().cuda(0)
optimizer = opt.SGD(net.parameters(), lr=0.001, momentum=0.9)
mom_res = train_model(net, optimizer, trainloader, verbose=False)
ploterr(mom_res)


# ### Nestorov Accelrated Gradient

# 1. Improves upon momentum. Momentum uses old gradient even at the new step
# 2. Nestorov uses the gradient at new point to update the gradient

# \begin{equation*}
# Momentum Update\\
#  w_t = w_{t-1} - \gamma * v_{t-1} - \lambda * \nabla J(w) \\
# \end{equation*}

# \begin{equation*}
# Momentum Update\\
#  v_t = \gamma * v_{t-1} + \lambda * \nabla J(w - \gamma * v_{t-1}) \\
#  w_t = w_{t-1} - v_t
# \end{equation*}

# In[ ]:


net = CifarNet().cuda(0)
optimizer = opt.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)
nest_res = train_model(net, optimizer, trainloader, verbose=False)
ploterr(nest_res)


# In[ ]:


Image("../input/optimization-talk-images/Nestorov.jpeg")


# ### AdaGrad

# 1. Main problem with the gradient descent is that learning rate is fixed and not adapted to the learning process
# 2. Main purpose of Adagrad is that it adapts learning rate to the parameters, performing smaller updates for frequently occurring values and larger updates for the sparse values
# 3. Because of this, it will well suited for the sparse data
# 4. The accumulated gradient in the denominator is one of the big problems where lr vanishes as learning goes on

# \begin{equation*}
# Momentum Update\\
#  gi = \sqrt{\sum_{k=1}^t v_k^2 + \epsilon} \\
#  w_t = w_{t-1} - \lambda / gi * \nabla J(w) \\
# \end{equation*}

# In[ ]:


net = CifarNet().cuda(0)
optimizer = opt.Adagrad(net.parameters(), lr=0.001)
adag_res = train_model(net, optimizer, trainloader, verbose=False)
ploterr(adag_res)


# ### RMSProp

# 1. A continuation of the AdaGrad. 
# 2. Instead of taking a normal sum of the squared gradients it calculates a decayed sum to handle the exploding denominator

# \begin{equation*}
# Update\\
#  eg_t = \sqrt{\gamma * eg_{t-1}^2 + (1-\gamma) * g_t^2} \\
#  w_t = w_{t-1} - \lambda / eg_t * \nabla J(w) \\
# \end{equation*}

# In[ ]:


net = CifarNet().cuda(0)
optimizer = opt.RMSprop(net.parameters(), lr=0.001, alpha=0.9, momentum=0.9)
rms_res = train_model(net, optimizer, trainloader, verbose=False)
ploterr(rms_res)


# ### Adam - Adaptive Moment Estimation

# 1. Similar to Adagrad and RMSProp adapts learning rates for parameters
# 2. Along with what RMSProp does it also stores the decaying average of past gradients, it also keeps track of the momentum and uses momentum for updation of the parameters
# 3. Beta1 is generally set to 0.9, beta 2 to 0.999 and epsilon to 1e-8

# \begin{equation*}
# Update\\
#  m_t = \beta_1 * m_{t-1} + (1-\beta_1) * g_t \\
#  v_t = \beta_2 * v_{t-1} + (1-\beta_2) * g_t^2 \\
#  w_t = w_{t-1} - \lambda/\sqrt{v_t+\epsilon} * m_t
# \end{equation*}

# In[ ]:


net = CifarNet().cuda(0)
optimizer = opt.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999))
adam_res = train_model(net, optimizer, trainloader, verbose=False)
ploterr(adam_res)


# ## Comparision

# In[ ]:


Image("../input/optimization-talk-images/contours_evaluation_optimizers.gif")


# ## References

# 1. http://ruder.io/optimizing-gradient-descent/index.html#batchgradientdescent
# 2. https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f
# 3. https://towardsdatascience.com/neural-network-optimization-algorithms-1a44c282f61d
# 4. https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/04-secondOrderOpt.pdf
# 5. http://cs231n.github.io/optimization-1/
# 6. https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c
