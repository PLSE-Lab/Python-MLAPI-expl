#!/usr/bin/env python
# coding: utf-8

# # What is Gradient Descent?
# [Index](https://www.kaggle.com/veleon/in-depth-look-at-machine-learning-models)
# 
# Gradient Descent is a generic optimization algorithm for minimizing cost functions by tweaking parameters iteratively. Gradient Descent is not useful for every Machine Learning model or metric. When computing the Mean Squared Error of a Linear Regression model you only have a global minimum. Not every model or metric has only one global minimum. A lot have multiple local minimums and one global minimum. Gradient Descent is used to find these local minimums and hopefully the global minimum as well.
# 
# ## What does it do?
# The Gradient Descent algorithm works by calculating the slope of a value in the desired cost function that you want to minimize. It then goes down this slope towards the lowest point. When it reaches this point it stops. Picture a ball rolling down a valley, it'll keep rolling toward the bottom but eventually it will be at the lowest point possible and it'll no longer move from its place.

# In[ ]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://cdn-images-1.medium.com/max/1600/0*rBQI7uBhBKE8KT-X.png")


# ## How does it calculate this?
# When implementing Gradient Descent there are 3 types of Gradient Descent you can choose from:
# * Batch Gradient Descent
# * Stochastic Gradient Descent
# * Mini-batch Gradient Descent
# 
# Each type of Gradient Descent calculates its minimum in its own way. So we'll go over all three of them in this kernel.
# 
# ## Creating a Dataset
# Before we start creating our Gradient Descent functions we'll have to create a dataset that we can use. To do this we'll use numpy's random library.
# 
# We'll use the formula:
# $y = 2+X

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
X = np.random.rand(100,1)
y = 2 + X+np.random.randn(100,1)/7.5
y_no_noise = 2 + X


# In[ ]:


plt.plot(X, y, 'ro')
plt.plot(X, y_no_noise, 'go')
plt.show()


# ## Batch Gradient Descent
# Batch Gradient Descent works by calculating the gradient vector ($\nabla_\theta$) of a cost function. This vector contains all _partial derivatives_ of the cost function. A _partial derivative_ is the value you get by calculating how much the cost function will change if you change parameter $\theta_j$ a tiny bit. For this Kernel we'll use the Mean Squared Error as our cost function.
# 
# Calculating the partial derivatives of the cost function for instance $i$:
# 
# $\frac{\partial}{\partial\theta_j}MSE(\theta) = \frac{2}{m}\displaystyle\sum_{i=1}^{m}(\theta^Tx^{(i)}-y^{(i)})x^{(i)}_j$
# 
# Where:
# * $\frac{\partial}{\partial\theta_j}MSE(\theta)$ is the partial derivative of the MSE cost function for parameter $j$ 
# * $\theta$ is the vector of parameters
# * $m$ is the number of instances in the dataset
# * $x$ is the vector of feature values for the  $i^{th}$ instance
# * $y$ is the label value for the  $i^{th}$ instance
# 
# 
# Calculating the Gradient Vector of a cost function:
# 
# $\nabla_\theta MSE(\theta)= \begin{bmatrix} \frac{\partial}{\partial\theta_0}MSE(\theta) \\ \frac{\partial}{\partial\theta_1}MSE(\theta) \\ \vdots \\ \frac{\partial}{\partial\theta_n}MSE(\theta) \end{bmatrix} = \frac{2}{m}X^T(X\theta-y)$
# 
# Where:
# * $\nabla_\theta MSE(\theta)$ is the gradient vector for the MSE cost function for all partial derivatives of $\theta$
# * $X$ is the matrix combining all feature values (excluding labels) of all instances in the dataset
# 
# Each Batch Gradient Descent step uses the full training data (the whole _batch_). This means that the Gradient Vector has to be calculated every step when using this algorithm.
# 
# Once you have calculated the Gradient Vector, which points uphill, you need to go downhill. You can do this by subracting the Gradient Vector from $\theta$. To make sure the size of the steps isn't to big we use the _learning rate_ $\eta$ (eta). The smaller our learning rate, the smaller our steps.
# 
# Calculating the Gradient Descent Step:
# 
# $\theta^{(next step)} = \theta - \eta \nabla_\theta MSE(\theta)$
# 
# Where:
# * $\theta^{(next step)}$ is the value of theta for the next step/iteration
# * $\eta$ is the learning rate

# In[ ]:


def calculateGradientVector(X, y, theta):
    X_b = np.c_[np.ones((len(X),1)), X] # concatenate a 1 to each instance (because we dont have x_0 in X)
    return (2/len(X))*X_b.T.dot(X_b.dot(theta)-y)

def batchGradientDescent(X, y, eta, iterations):
    np.random.seed(42)
    theta = np.random.randn(2,1)
    for i in range(iterations):
        gradientVector = calculateGradientVector(X, y, theta)
        theta = theta - eta * gradientVector
    return theta

batchGradientDescent(X,y,0.1,1000)


# As we can see, the Batch Gradient Descent gives us the same values for $\theta_0$ and $\theta_1$ as the Normal Equation that we used in the Linear Regression Kernel. This would be the optimal outcome for this dataset. But what would happen if we changed the _learning rate_?
# 
# 
# ## Learning Rate
# To show the effect of _learning rate_ on Gradient Descent we'll plot the first "guesses" of the algorithm. This will show us how big the steps are that the algorithm makes. We'll also give the first and last guess a different color so we can distinguish them from all the other lines.

# In[ ]:


def predictY(x, theta): # predicts a single y value
    return theta[0]+theta[1]*x

def getLearningRatePlot(X, y, eta, iterations):
    plt.plot(X, y, 'ro')
    for i in range(iterations):
        theta = batchGradientDescent(X, y, eta, i)
        if i is 0:
            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'r--')
        elif i is iterations-1:
            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'g-' , linewidth=3)
        else:
            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'b-')
    plt.xlabel('$X_1$', fontsize=20)
    plt.title("$\eta = {}$ for ${}$ iterations".format(eta, iterations), fontsize=20)
    plt.axis([0, 1, 0, 4])


# In[ ]:


plt.figure(figsize=(20,4))
plt.subplot(131); getLearningRatePlot(X, y, 0.02, 10)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132); getLearningRatePlot(X, y, 0.1, 10)
plt.subplot(133); getLearningRatePlot(X, y, 0.8, 10)


# As you can see in the 3 different plots, the _learning rate_ has a big effect on the $\theta$ output of Gradient Descent. When the _learning rate_ is too small (the first plot) then it'll take only small steps. This aproach gets at the optimal $\theta$ eventually but it'll take a really long time. When you have a good _learning rate_ (the second plot) you take some big steps toward the optimal $\theta$ at first but then take smaller steps as you get closer. Last, when your _learning rate_ is too high (the third plot) it'll overshoot the target. It will slowly close in but will most likely never reach the optimal $\theta$ or at least take a really long time. We'll plot all plots once more but this time we'll add some more iterations. 

# In[ ]:


plt.figure(figsize=(20,4))
plt.subplot(131); getLearningRatePlot(X, y, 0.02, 100)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132); getLearningRatePlot(X, y, 0.1, 100)
plt.subplot(133); getLearningRatePlot(X, y, 0.8, 100)


# The low _learning rate_ takes a while to get at the optimal $\theta$ but it did get there. The good _learning rate_ shows us that it gets near to the optimum fast and then starts slowly optimizing it for the optimal $\theta$. When the high _learning rate_ goes on for a little longer it'll slowly converge on the optimal $\theta$ but right now it hasn't even come close.
# 
# ## Stochastic Gradient Descent
# The problem with Batch Gradient Descent is the use of the full training set every step. This takes a long time to compute and thus makes it very slow with large training sets. Stochastic Gradient Descent is the complete opposite of this. It picks a random instance in the training set every step and computes the gradients based on this instance. This increases the speed of the algorithm immensely. This also makes it possible to train on huge datasets because it doesn't use all the data at once.
# 
# The random nature of the Stochastic Gradient Descent algorithm does cause some problems. It doesn't gently decrease until it reaches a minimum, instead it goes up and down while decreasing on average. This means that once it reaches its optimal $\theta$ it will still move around. Because of this SGD will only get a very good $\theta$ but (almost) never the optimal $\theta$.
# 
# When cost function is irregular (containing multiple local minima and one global minimum) this can help it "jump out" of the local minima towards the global minimum so SGD does have a better chance at finding these then the BGD does. 
# 
# So randomness is a good thing when you want to find the global minimum but a bad thing when you want to get the actual global minimum. How do we fix this? We use a _learning schedule_, a function that decreases our step size when we get to a higher iteration.
# 
# A function that can compute Stochastic Gradient Descent will have to exist out of these parts:
# * Random instance selection
# * A learning schedule to calculate eta
# * Gradient Vector calculation (of the instance)

# In[ ]:


def learningSchedule(t, t0=5, t1=50):
    #t0 and t1 define your starting eta (5/50 = 0.1) and the growth rate (1/11 = 0.090 but 5/51 = 0.098)
    return t0 / (t+t1)

def stochasticGradientDescent(X, y, epochs, t0=5, t1=50):
    np.random.seed(42)
    theta = np.random.randn(2,1)
    for epoch in range(epochs):
        for i in range(len(X)):
            random_iteration = np.random.randint(len(X))
            x_i = X[random_iteration:random_iteration+1]
            y_i = y[random_iteration:random_iteration+1]
            gradientVector = calculateGradientVector(x_i, y_i, theta)
            eta = learningSchedule(epoch*len(X)+i, t0, t1)
            theta = theta - eta*gradientVector
    return theta

stochasticGradientDescent(X, y, 1000)


# In[ ]:


def plotStochastic(X,y,iterations, t0=5, t1=50): 
    plt.plot(X, y, 'ro')
    for i in range(iterations):
        theta = stochasticGradientDescent(X, y, i, t0, t1)
        if i is 0:
            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'r--')
        elif i is iterations-1:
            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'g-' , linewidth=3)
        else:
            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'b-')
    plt.xlabel('$X_1$', fontsize=20)
    plt.title("SGD for ${}$ iterations with schedule {}/{}".format(iterations,t0,t1), fontsize=15)
    plt.axis([0, 1, 0, 4])


# In[ ]:


plt.figure(figsize=(20,4))
plt.subplot(131); plotStochastic(X, y, 10)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132); plotStochastic(X, y, 10, 1, 100)
plt.subplot(133); plotStochastic(X, y, 10, 1, 200)


# That's pretty close to the BGD $\theta$! BDG will be superior on this dataset because there is only one minimum but this just shows how close SGD can get. Now that we have seen both Batch and Stochastic Gradient Descent there is only one more Gradient Descent type left.
# 
# ## Mini-Batch Gradient Descent
# What if we combined Batch Gradient Descent and Stochastic Gradient Descent? We would get Mini-Batch Gradient Descent. Using a small random subset of the training set (a _mini-batch_) to compute the Gradient Vector creates less randomness than SGD has and still has the possibility to "jump out" of local minima, though not as much as SGD.

# In[ ]:


def miniBatchGradientDescent(X, y, epochs, batchSize, t0=5, t1=50):
    np.random.seed(42)
    theta = np.random.randn(2,1)
    for epoch in range(epochs):
        for i in range(len(X)):
            x_i = np.zeros((batchSize,1))
            y_i = np.zeros((batchSize,1))
            for b in range(batchSize):
                random_iteration = np.random.randint(len(X))
                x_i[b] = X[random_iteration]
                y_i[b] = y[random_iteration]
            gradientVector = calculateGradientVector(x_i, y_i, theta)
            eta = learningSchedule(epoch*len(X)+i, t0, t1)
            theta = theta - eta*gradientVector
    return theta

miniBatchGradientDescent(X, y, 1000, 20)


# In[ ]:


def plotMBGD(X,y,iterations, batchSize): 
    plt.plot(X, y, 'ro')
    for i in range(iterations):
        theta = miniBatchGradientDescent(X, y, i, batchSize, t0=1, t1=100)
        if i is 0:
            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'r--')
        elif i is iterations-1:
            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'g-' , linewidth=3)
        else:
            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'b-')
    plt.xlabel('$X_1$', fontsize=20)
    plt.title("MBGD for ${}$ iterations with batchsize {}".format(iterations,batchSize), fontsize=15)
    plt.axis([0, 1, 0, 4])
    
plt.figure(figsize=(20,4))
plt.subplot(121); plotStochastic(X, y, 10, 1, 100)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(122); plotMBGD(X, y, 10, 10)


# ## Conclusion
# Now that we have tried all 3 types of Gradient Descent you hopefully have a better grasp of when you can best use which type of Gradient Descent. If you want to use Gradient Descent without creating your own function then Scikit-Learn has its _linearmodel.SGDRegressor_ model. Sadly it doesn't have Batch Gradient Descent or Mini-Batch Gradient Descent.
# 
# ### Previous Kernel
# [How Does Linear Regression Work?](https://www.kaggle.com/veleon/how-does-linear-regression-work)
# ### Next Kernel
# [Polynomial Regression](https://www.kaggle.com/veleon/what-is-polynomial-regression)
