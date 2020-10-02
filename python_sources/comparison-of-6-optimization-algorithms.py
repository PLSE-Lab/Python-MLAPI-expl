#!/usr/bin/env python
# coding: utf-8

# # Comparison of 6 optimization algorithms used in deep learning
# 
# The aim of this notebook is to compare the performance of six different optimization algorithms used in deep learning. This algorithms are the following ones:
# * Gradient descent optimization
# * Momentum optimization
# * Nesterov accelerated gradient optimization
# * AdaGrad optimizer
# * RMSProp optimizer
# * Adam optimization

# Their performance will be tested in a simple linear regression, so as to visualizing the obtained results
# 
# As you may know, the linear regression model is the shown below:

# In[ ]:


from IPython.display import display, Math, Latex

display(Math(r'y_p = \theta\cdot X'))


# Where:
# * theta is the model's parameter vector, containing the bias term. In this case, theta is a column vector containing two values, the bias term and the slope term
# * X is the feature vector, containing a x0 feature equal to 1 (multiplied to the bias term) and a x1 term
# * yp is the prediction of the model

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# To generate the data set, a linear model of parameters 5 (slope) and 4 (bias) will be simulated, and then it will be aded some noise with a normal distribution of 0 mean and 0.2 deviation

# In[ ]:


np.random.seed(42)
m = 1000
n = 2
x = np.linspace(0, 1, m)
ones = np.ones(m)
X = np.column_stack((ones, x))
Y = 5 * x + 4
y = 5 * x + 4 + np.random.randn(len(x))/5
y = y.reshape(m, 1)


# In[ ]:


labels = ['Instances', 'Initial model']

plt.plot(x, y, 'go')
plt.plot(x, Y, 'r')
plt.legend(labels)
plt.show()


# Some initializations that will be used later:

# In[ ]:


np.random.seed(42)
theta0 = np.array([np.random.randn(), np.random.randn()])
theta0 = theta0.reshape(n, 1)

m0 = np.array([0, 0])
m0 = m0.reshape(n, 1)

s0 = np.array([0, 0])
s0 = s0.reshape(n, 1)


# To compute the cost of the model, it has been used the mean square error function shown below:

# In[ ]:


display(Math(r'MSE(\theta) = \frac{1}{m} \cdot \sum_{i=1}^{m}(X^{(i)} \theta - y^{(i)})^2'))


# Which has the following gradient:

# In[ ]:


display(Math(r'\nabla_\theta MSE(\theta) = \frac{2}{m} X^T (X\theta - y)'))


# This functions are created in the cell below:

# In[ ]:


mse = lambda x: (1/m) * np.sum((X @ x - y) ** 2)

grad_mse = lambda x: (2/m) * (X.T @ (X @ x - y))


# So as to measure the optimization algorithms performance, the same learning rate and max. num of epochs will be used in all of them:

# In[ ]:


learning_rate = 0.005
max_epochs = 100000


# In[ ]:


thetas = []
_losses = []
n_epochs = []


# ## Normal equation

# As we are using linear regression, there is a mathematical solution that solves the problem:

# In[ ]:


display(Math(r'\theta = (X^T X)^{-1}X^T y'))


# In[ ]:


mathematical_solution = np.linalg.inv(X.T @ X) @ X.T @ y
mathematical_solution


# ## Gradient descent optimization
# 
# The first algorithm we will code is the gradient descent, that works as follows:

# In[ ]:


display(Math(r'\theta_{next step} = \theta - \alpha \nabla_\theta MSE(\theta)'))


# Where:
# * alpha = learning rate
# 
# Every 20 epochs the loss will be evaluated, and if it's difference with the loss 20 epochs before is lower than 10exp(-5), actual values of theta will be taken as good, so we will stop iterating. If not, we will continue with iterations

# In[ ]:


def gradient_descent(epochs, learning_rate, loss_variation=10**-5):
    
    theta = theta0
    losses = []
    last_loss = 9999
    
    for epoch in range(epochs):
        loss = mse(theta)
        gradients = grad_mse(theta)
        theta = theta - (learning_rate * gradients)
        
        if epoch % 20 == 0:
            losses.append((loss, epoch))
            if(abs(last_loss - loss) < loss_variation):
                losses = np.array(losses)
                return theta, losses, epoch
            else:
                last_loss = loss
    
    losses = np.array(losses)
    print('Max. number of iterations without converging')   
    return theta, losses, epochs


# In[ ]:


p1, losses, epoch = gradient_descent(max_epochs, learning_rate)
thetas.append(p1)
_losses.append(losses)
n_epochs.append(epoch)


# Let's see the learning curve:

# In[ ]:


plt.plot(_losses[0][0:, 1], _losses[0][0:, 0])
plt.show()


# Due to the fact that theta is randomly initialized, the mse in the first epochs is very high, so the learning curve is not very clear. Thus, we will zoom it starting in the 1000th epoch:

# In[ ]:


plt.plot(_losses[0][50:, 1], _losses[0][50:, 0])
plt.show()


# ## Momentum optimization
# 
# Gradient descent simply updates the weights theta substracting the gradient of the cost function multiplied by the learning rate. It doesn't care about what the earlier gradients were. Thus, if the local gradient is small, it goes very slowly. Momentum optimization cares about the previous gradient, however. At each iteration, it updates the momentum vector with the gradients, what it means that gradients are used as acceleration, not as speed. The algorithm is shown below:

# In[ ]:


display(Math(r'1.\quad m \leftarrow \beta m - \alpha \nabla_\theta MSE(\theta)'))
display(Math(r'2.\quad \theta = \theta + m'))


# In[ ]:


def momentum_optimization(epochs, learning_rate, beta, loss_variation=10**-5):
    
    theta = theta0
    losses = []
    m = m0
    last_loss = 9999
    
    for epoch in range(epochs):
        loss = mse(theta)
        gradients = grad_mse(theta)
        m = beta * m + learning_rate * gradients
        theta = theta - m
        
        if epoch % 20 == 0:
            losses.append((loss, epoch))
            if(abs(last_loss - loss) < loss_variation):
                losses = np.array(losses)
                return theta, losses, epoch
            else:
                last_loss = loss
            
    losses = np.array(losses)
    print('Max. number of iterations without converging')  
    return theta, losses, epochs


# In[ ]:


p1, losses, epoch = momentum_optimization(max_epochs, learning_rate, 0.9)
thetas.append(p1)
_losses.append(losses)
n_epochs.append(epoch)

plt.plot(_losses[1][:, 1], _losses[1][:, 0])
plt.show()


# Let's zoom as we did in gradient descent:

# In[ ]:


plt.plot(_losses[1][3:, 1], _losses[1][3:, 0])
plt.show()


# ## Nesterov accelerated gradient optimization
# 
# It is slightly different to momentum optimization: it measures the gradient not in the local position, but in slightly ahead in the direction of the momentum:

# In[ ]:


display(Math(r'1.\quad m \leftarrow \beta m - \alpha \nabla_\theta MSE(\theta + \beta m)'))
display(Math(r'2.\quad \theta = \theta + m'))


# In[ ]:


def nesterov_accelerated_gradient(epochs, learning_rate, beta, loss_variation=10**-5):
    
    theta = theta0
    losses = []
    m = m0
    last_loss = 9999
    
    for epoch in range(epochs):
        loss = mse(theta)
        gradients = grad_mse(theta + beta * m)
        m = beta * m + learning_rate * gradients
        theta = theta - m
        
        if epoch % 20 == 0:
            losses.append((loss, epoch))
            if(abs(last_loss - loss) < loss_variation):
                losses = np.array(losses)        
                return theta, losses, epoch
            else:
                    last_loss = loss
            
    losses = np.array(losses)
    print('Max. number of iterations without converging')
    return theta, losses, epochs


# In[ ]:


p1, losses, epoch = nesterov_accelerated_gradient(max_epochs, learning_rate, 0.9)

thetas.append(p1)
_losses.append(losses)
n_epochs.append(epoch)

plt.plot(_losses[1][:, 1], _losses[1][:, 0])
plt.show()


# Zooming it:

# In[ ]:


plt.plot(_losses[1][3:, 1], _losses[1][3:, 0])
plt.show()


# ## AdaGrad optimization
# 
# Gradient descent starts by quickly going down the steepest slope, and then slowly goes down to the bottom of the valley. The AdaGrad algorithm detects this and corrects it's direction to point more toward the global optimum, scalling down the gradient vector along the steepest dimentions. This leads to a decay of the learning rate, but it dos as fast in steep dimentions as in gentler slopes.

# In[ ]:


display(Math(r'1.\quad s \leftarrow s + \nabla_\theta MSE(\theta)\otimes \nabla_\theta MSE(\theta)'))
display(Math(r'2. \quad \theta \leftarrow \theta - \alpha \nabla_\theta MSE(\theta) \oslash \sqrt{s + \epsilon}'))


# In[ ]:


def adaGrad(epochs, learning_rate, epsilon=10**-10, loss_variation=10**-5):
    
    theta = theta0
    losses = []
    s = s0
    last_loss = 9999
    
    for epoch in range(epochs):
        loss = mse(theta)
        gradients = grad_mse(theta)
        s = s + gradients * gradients
        theta = theta - (learning_rate * gradients) / (np.sqrt(s+ epsilon))
        
        if epoch % 20 == 0:
            losses.append((loss, epoch))
            if (abs(last_loss - loss) < loss_variation):
                losses = np.array(losses)
                return theta, losses, epoch
            else:
                last_loss = loss
            
    losses = np.array(losses)
    print('Max. number of iterations without converging')
    return theta, losses, epochs


# In[ ]:


p1, losses, epoch = adaGrad(max_epochs, learning_rate)

thetas.append(p1)
_losses.append(losses)
n_epochs.append(epoch)

plt.plot(_losses[3][:, 1], _losses[3][:, 0])
plt.show()


# As you may notice, this algorithm needs more iterations to converge, due to the fact that theta0 is randomly initialized, so it is very far away the optimum solution. As the learning rate decays, the algorithm does not have time to reach the solution

# ## RMSProp optimizer
# 
# Owing to the fact that AdaGrad slows down too fast, RMSProp algoithm only accumulates the gradients from the most recent iterations, by using exponential decay:

# In[ ]:


display(Math(r'1.\quad s \leftarrow \beta s + (1 - \beta) \nabla_\theta MSE(\theta)\otimes \nabla_\theta MSE(\theta)'))
display(Math(r'2. \quad \theta \leftarrow \theta - \alpha \nabla_\theta MSE(\theta) \oslash \sqrt{s + \epsilon}'))


# In[ ]:


def RMSProp(epochs, learning_rate, beta, epsilon=10**-10, loss_variation=10**-5):
    
    theta = theta0
    losses = []
    s = s0
    last_loss = 9999
    
    for epoch in range(epochs):
        loss = mse(theta)
        gradients = grad_mse(theta)
        s = beta * s + (1 - beta) * gradients * gradients
        theta = theta - (learning_rate * gradients) / (np.sqrt(s + epsilon))
        if epoch % 20 == 0:
            losses.append((loss, epoch))
            if(abs(last_loss - loss) < loss_variation):
                losses = np.array(losses)
                return theta, losses, epoch
            else:
                last_loss = loss
            
    losses = np.array(losses)
    print('Max. number of iterations without converging')    
    return theta, losses, epochs


# In[ ]:


p1, losses, epoch = RMSProp(max_epochs, learning_rate, 0.9)

thetas.append(p1)
_losses.append(losses)
n_epochs.append(epoch)

plt.plot(_losses[4][:, 1], _losses[4][:, 0])
plt.show()


# ## Adam optimization
# 
# This algorithm combines the ideas of the momentum optimization and RMSProp:
# * keeps track of an exponentially decaying average of past gradients
# * keeps track of an exponentially decaying average of past squared gradients

# In[ ]:


display(Math(r'1. \quad m \leftarrow \beta_1 m - (1 - \beta_1)\nabla_\theta MSE(\theta)'))
display(Math(r'2. \quad s \leftarrow \beta_2 s + (1 - \beta_2)\nabla_\theta MSE(\theta) \otimes\nabla_\theta MSE(\theta)'))
display(Math(r'3. \quad \hat{m} \leftarrow \frac{m}{1 - \beta_1^T}'))
display(Math(r'4. \quad \hat{s} \leftarrow \frac{s}{1 - \beta_2^T}'))
display(Math(r'5. \quad \theta \leftarrow \theta + \alpha \hat{m} \oslash \sqrt{\hat{s} + \epsilon}'))
print('Where T represents the iteration number')


# In[ ]:


def adam_opt(epochs, learning_rate, beta1, beta2, epsilon=10**-10, loss_variation=10**-5):
    
    theta = theta0
    losses = []
    s = s0
    m = m0
    last_loss = 9999
    
    for epoch in range(epochs):
        e = epoch
        loss = mse(theta)
        gradients = grad_mse(theta)

        m = beta1 * m + (1 - beta1) * gradients
        s = beta2 * s + (1 - beta2) * gradients * gradients

        m2 = m / (1 - beta1**(epoch+1))
        s2 = s / (1 - beta2**(epoch+1))

        theta = theta - (learning_rate * m2 )/ (np.sqrt(s2 + epsilon))

        if epoch % 20 == 0:
            losses.append((loss, epoch))
            if(abs(last_loss - loss) < loss_variation):
                losses = np.array(losses)
                return theta, losses, epoch
            else:
                last_loss = loss
        
    losses = np.array(losses)
    print('Max. number of iterations without converging') 
    return theta, losses, epochs


# In[ ]:


p1, losses, epoch = adam_opt(max_epochs, learning_rate, 0.9, 0.9)

thetas.append(p1)
_losses.append(losses)
n_epochs.append(epoch)

plt.plot(_losses[5][:, 1], _losses[5][:, 0])
plt.show()


# ## Results

# ### Needed iterations to reach the optimum solution

# In[ ]:


labels = ['Gradient Descent', 'Momentum', 'Nesterov', 'AdaGrad', 'RMSProp', 'Adam']
y_pos = np.arange(len(n_epochs))

fig = plt.figure(figsize=(15,6))
plt.bar(y_pos, n_epochs)
plt.xticks(y_pos, labels)
plt.show()


# Owing to the fact that AdaGrad has not converged, we'll remove it from the visualization, so as to have a clearer one

# In[ ]:


labels.remove('AdaGrad')
n_epochs = np.delete(n_epochs, 3)
y_pos = np.arange(len(n_epochs))

fig = plt.figure(figsize=(15,6))
plt.bar(y_pos, n_epochs)
plt.xticks(y_pos, labels)
plt.show()


# As you might see, Gradient Descent is much slower than the rest of algorithms. Momentum and Nesterov are slightly faster due to the fact that they don't have adaptative learning rate, like RMSProp and Adam have. However, this result does not mean that Momentum and Nesterov are always faster than RMSProp and Adam, it depends both in the shape of the cost function and the initialization of theta.

# ### Obtanied model

# In[ ]:


fig = plt.figure(figsize=(15,6))
# Instances
plt.plot(x, y, 'go')
# First model
plt.plot(x, Y, 'r')
# Mathematical solution
ms = mathematical_solution[0] + mathematical_solution[1] * x
plt.plot(x, ms, 'y')
# Gradient descent
gd = thetas[0][0] + thetas[0][1] * x
plt.plot(x, gd, 'b')
# Momentum
mo = thetas[1][0] + thetas[1][1] * x
plt.plot(x, mo, 'cyan')
# Nesterov
no = thetas[2][0] + thetas[2][1] * x
plt.plot(x, no, 'salmon')
# AdaGrad
ag = thetas[3][0] + thetas[3][1] * x
plt.plot(x, ag, 'black')
# RMSProp
rms = thetas[4][0] + thetas[4][1] * x
plt.plot(x, rms, 'orange')
# Adam
adam = thetas[4][0] + thetas[4][1] * x
plt.plot(x, adam, 'purple')

labels = ['Instances', 'Initial model', 'Mathematical solution', 'Gradient descent',
         'Momentum optimizer', 'Nesterov optimizer', 'AdaGrad optmizer', 'RMSProp optimizer', 'Adam optimizer']

plt.legend(labels)

plt.show()


# As shown above, all models has reached a the optimal solution except the AdaGrad optimizer, which would need more iterations or a higher learning rate to reach the solution
