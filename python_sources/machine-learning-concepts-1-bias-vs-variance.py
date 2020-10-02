#!/usr/bin/env python
# coding: utf-8

# ***Hey! We will discuss Bias and Variance of ML Algorithms/ Models***
# 
# 
# When we create a machine learning model, we usually encounter the below issues:
# 
# 1. Our model does not fit the data (training data) that well, may be because it is not flexible enough 
# 
# or 
# 
# 2. Our model does very well (i.e it is very accurate) on  the (training) data but doesnt perform too well when used on new (or validation) data
# 
# 
# 
# 
# 

# 
# The first situation is where bias is high or the model is underfitting. In simpler terms model does not capture the all the relevant relations that well and hence has significant error. This happens in simpler and parametric modles like linear regression. (A simple line may not be so good approximation of a curve.its just not that flexible!) 
# 
# ![image.png](attachment:image.png)
# 
# 

# The second situation is where variance is high or the model is over fitting. It is very good at caputuring all the details in the data we trained it with but performs bad on ne data ( may be it modeled the noise too). This happens usually in case of more complex models like neural nets.
# 
# ![image.png](attachment:image.png)
# 

# The same can be visualized as :
# 
# ![image.png](attachment:image.png)

# There should be a tradeoff between bias and varaince to arrive at the best model.
# 
# ![image.png](attachment:image.png)
# 
# 
# 

# The same can be interpreted as :
# 
# ![image.png](attachment:image.png)

# **How do we fix high bias or high variance in the data set?**
# 
# High bias is due to a simple model and we also see a high training error. To fix that we can do following things:
# 
# 1. Add more input features
# 2. Add more complexity by introducing polynomial features
# 3. Decrease Regularization term
# 
# 
# 
# ![image.png](attachment:image.png)

# High variance is due to a model that tries to fit most of the training dataset points and hence gets more complex. To resolve high variance issue we need to work on
# 1. Getting more training data
# 2. Reduce input features
# 3. Increase Regularization term
# 
# 
# ![image.png](attachment:image.png)

# Before we wind up the topic of bias and variance, a brief on Regularization.
# 
# Regularization is a technique where we penalize the loss function for a complex model which is very flexible. This helps with overfitting. It does this by penalizing the different parameters or weights to reduce the noise of the training data and generalize well on the test data.
# 
# 
# Regularization significantly reduces the variance without substantially increasing bias.
# 
# 
# 

# Some sources:
# 
# 
# https://medium.com/datadriveninvestor/bias-and-variance-in-machine-learning-51fdd38d1f86
# https://missinglink.ai/guides/neural-network-concepts/neural-network-bias-bias-neuron-overfitting-underfitting/
# https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229
# https://community.alteryx.com/t5/Data-Science-Blog/Bias-Versus-Variance/ba-p/351862
