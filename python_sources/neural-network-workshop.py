#!/usr/bin/env python
# coding: utf-8

# # Building a Neural Network From Scratch
# ## Today we are going to build a neural network from scratch using only numpy: a linear algebra library
# But before that. Lets take a moment to understand how a neural network works. A neural network can be thought of as a function approximator, a point estimator, a conditional probability distribution and many more!. The takeaway is that there a multiple angles we can view the idea of a neural network: computer science, neuroscience, cognitive science, statistics and probability. But to build one, we need to be concerned with one viewpoint: multivariate calculus. 
# 
# The goal of this neural network will be to classify handwritten digits. Pretty neato burrito huh? For now lets import the libraries we will be using as well as prepare our data for learning.

# In[ ]:


#Import libraries

#Load data


#Prepare Data

#Prepare Labels

#Showcase the data


# ## Ok, what is a neural network?
# We can think of a neural network as a layered function that learns to do a *task*. Lets begin by first defining our functions.
# $h(x) = l_n(l_{n-1}(\dots l_1(l_0(x))))$ is the neural network function. Note that our function is a composition of layer functions 
# 
# $l_n(z) = \sigma(xW+b)$ is the layer function. There are a lot of moving parts to this so lets break it down.
# 
# $\sigma(z)$ is a non-linearity function. This allows our network to learn non linear functions!
# 
# $z = xW + b$ sometimes we like to define a z term for convenience. The z term is a matrix multiplication between an input x and a weight matrix W and a linear shift by a bias vector b (note a bias is not always necessary and can be omitted). We learn the weight matrix and bias vector to perform the *task*. This is what we will be fucking with!
# 
# $h(x) = \sigma(\sigma \dots(\sigma(\sigma(xW_0 + b_0)W_1 + b_1)\dots W_{n-1} + b_{n-1})W_n + b_n)$ Putting it all together we can see a neural network is nothing but a *chain* of successive matrix multiplications, linear shifts, and non-linearity functions. 
# 
# Typically we have different weight matrixes (of different sizes), no bias and a one to two types of non-lineary function, but exceptions exist to all three conditions! There is a lot of cool fun and creative things one can do to build a neural network. We call the actual techniques, shape, and functions that compose a neural network its *architecture*.

# ## Our Neural Network
# Today our architecture will be follow the shape 784 x 16 x 16 x 10. This is the same architecture that is used in 3blue1brown's video. So lets initialize our learnable parameters!

# In[ ]:


#Initialize Weights


#Initialize Biases


# $\sigma(z) = \frac{1}{1 + e^{-z}}$ is called the sigmoid function and will be the non-linearity function we will use for our model.

# In[ ]:





# ## Cool. So how does the damn thing learn?
# So first what do we mean by learning a *task*? Clearly, telling a neural network to learn how to classify hand written digits won't work: our neural network speaks math and our computers speak binary. Therefore, we need to define *mathematically* what we mean by learning a task. Herein lies the problem: tasks that are trivial to you or me (like classifying hand written digits) are **impossible** to define mathematically. As a result we instead define a function called a **Loss** or **Cost** that serves as a proxy task we can use to learn from. A Cost function serves as an indirect method that helps us learn a task from. We therefore *optimize* our model on our Cost function, and then pray it does the task we want it to do. There is a huge variety of Cost functions and selecting an appropriate cost for an appropriate task is an area of active research. 
# 
# $C(y, h(x))$  is how we denote a supervised cost function. It is supervised because we have this y term. The y term is a label for a given data point x. Typically we try to predict what y will be (such as the number of a handwritten digit).
# 
# For our neural network we will be using a Cost function called *Mean Squared Error*
# 
# $MSE(y, h(x)) = \frac{1}{n}\Sigma(h(x) - y)^2$ is how Mean Squared Error is defined. What this optimizing this Cost function will do is minimize the Variance and Bias squared of our prediction. While Bias and Variance are terms outside the scope of this workshop, we can understand the idea of using this Cost function as *by minimizing Variance and Bias squared we will be indirectly improving how good our model is at predicting a handwritten digit*. Lets implement our cost function now.

# In[ ]:





# 
# ## Baby got Back...propagation
# Great. Now lets go back to that keyword *optimize*. To optimize a function is to find the minimum or maximum point (function depending) that achieves the best desired result. Since the learning of our neural network has become a optimization problem, we an use any optimization technique to *train* our neural network. The most common optimization method used is called **gradient descent**. Gradient Descent involves finding the gradient of a function, and moving along that gradient to either a minimum or maximum. Typically for neural network cost functions we want to minimize them (again, exceptions do exist). To do gradient descent we use an algorithm called **backpropagation**. Backpropagation is the algorithm that makes all this possible. We will be implementing backprogation together today. 
# 
# Lets take a look at the expanded function we are going to be optimizing.
# 
# $MSE(y, h(x)) = \frac{1}{n}\Sigma(\sigma(\sigma \dots(\sigma(\sigma(xW_0 + b_0)W_1 + b_1)\dots W_{n-1} + b_{n-1})W_n + b_n) - y)^2$
# 
# Whack. But to do gradient descent we need to find the gradient of the weights and bias (our learnable parameters). This is where our friend chain rule comes into play. Lets take it step by step. First off lets find the gradient of MSE with respect to h(x)
# $\frac{d}{dh(x)}MSE(y, h(x)) = \frac{d}{dh(x)}\frac{1}{n}\Sigma(h(x) - y)^2 \\
# \frac{d}{dh(x)}MSE(y, h(x)) = \frac{2}{n}\Sigma (h(x) - y)$ 
# 
# Simple enough, we can implement it right away!

# In[ ]:





# While getting the derivative of the Cost function is a start. It is certainly not enough. Recall that our neural network is a chain of successive matrix multiplications, linear shifts, and non-linearity functions. We are interested in finding the gradient with respect to the multiplications and linear shifts, and to do that we need to break through the non-linearity function. We are using the sigmoid activation function. Lets take the derivative of the sigmoid function with respect to our z term (notice how we are slowly breaking into our neural network function).
# 
# $\frac{d}{dz}\sigma(z) = \frac{d}{dz}\frac{1}{1 + e^{-z}} \\
# \frac{d}{dz}\sigma(z) = \frac{-1}{(1 + e^{-z})^2} \frac{d}{dz}(1+e^{-z}) \\
# \frac{d}{dz}\sigma(z) = \frac{1}{(1 + e^{-z})^2}e^{-z}$
# 
# Neat! But we can go even further beyond. 
# 
# $\frac{d}{dz}\sigma(z) = \frac{1}{(1 + e^{-z})^2}e^{-z} \\
# \frac{d}{dz}\sigma(z) = \frac{1}{(1 + e^{-z})} \frac{e^{-z}}{(1 + e^{-z})} \\
# \frac{d}{dz}\sigma(z) = \frac{1}{(1 + e^{-z})} \frac{e^{-z} + 0}{(1 + e^{-z})} \\
# \frac{d}{dz}\sigma(z) = \frac{1}{(1 + e^{-z})} \frac{e^{-z} + 1 - 1}{(1 + e^{-z})} \\
# \frac{d}{dz}\sigma(z) = \frac{1}{(1 + e^{-z})} (\frac{e^{-z} + 1}{(1 + e^{-z})}  - \frac{1}{(1 + e^{-z})} ) \\
# \frac{d}{dz}\sigma(z) = \frac{1}{(1 + e^{-z})} (\frac{1 + e^{-z}}{(1 + e^{-z})}  - \frac{1}{(1 + e^{-z})} ) \\
# \frac{d}{dz}\sigma(z) = \frac{1}{(1 + e^{-z})} (1  - \frac{1}{(1 + e^{-z})} ) \\
# \frac{d}{dz}\sigma(z) = \sigma(z)(1 - \sigma(z)) \\
# $
# 
# This is one of the reasons by the sigmoid activation is one of the most wildly used activation functions. It is simple to implement and its derivative is very pretty. uwu. Lets code it up
# 
# 

# In[ ]:





# Now, lets break apart that z term. Recall that the z term consists of a matrix multiplication, 

# In[ ]:





# In[ ]:





# ## Calculus Primer
# #### Power rule
# $\frac{d}{dx}x^n \rightarrow nx^{n-1}$
# #### Exponent rule
# $\frac{d}{dx}e^{ax} \rightarrow ae^{ax}$
# #### Log rule
# $\frac{d}{dx}ln(ax) \rightarrow \frac{a}{x}$
# #### Chain rule
# $\frac{d}{dx}f(h(x)) \rightarrow f'(x)\frac{d}{dx}h(x)$
# Keep an eye out for the chain rule! Its what makes the whole algorithm tick.

# ## Linear Algebra Primer
# #### Matrix Multiplication
# *Insert primer here*
# #### Matrix Transposition
# *Insert primer here*
