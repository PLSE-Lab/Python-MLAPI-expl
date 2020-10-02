#!/usr/bin/env python
# coding: utf-8

# Logistic Regression is a very popular technique borrowed from statistics to apply to machine learning application. It is the most popular method for classification problems. What is classification problem ?  Problems that have discrete outcomes (class values). Therefore a binary classification problem has only two outcomes.
# 
# Let's dive into the maths behind Logistic Regression.
# 
# Logistic Regression is named for the function used in the calculation - logistic function.
# 
# The logistic function is also called the Sigmoid function:

# <img src="https://isteam.wsimg.com/ip/96331c9c-69bc-4562-9ea2-648231e1d0b3/sigmoid-0001.JPG/:/rs=w:1280">

# where e is the base of the natural logarithms and x is the value that you want to transform. x can be a set of multiple independent variables. 
# 
# This is the plot of the numbers between 10 and -10 transformed into the range of values between 0 and 1:

# <img src="https://img1.wsimg.com/isteam/ip/96331c9c-69bc-4562-9ea2-648231e1d0b3/logistic_plot.JPG/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280">

# Logistic Regression is very much like Linear Regression where independent variables (the x's) are combined linearly with weights (the w's) to predict an output value y. The key difference between Logistic Regression and Linear Regression is that Logistic predict a binary outcome (0 or 1).
# 
# Logistic Regression models the probability of the outcome based on the independent variables (the x's) and the weights (the w's). And of course the predicted value has to be transformed into a crisp prediction, in most cases, if the probability is < 0.5, we draw on the conclusion that the probability is 0, on the other hand, if the probability is > 0.5 then the probability is 1. what about when it is = 0.5 ? it is all up to you.
# 
# The Logistic Regression model is stated as:

# <img src="https://img1.wsimg.com/isteam/ip/96331c9c-69bc-4562-9ea2-648231e1d0b3/logistic_2.JPG/:/rs=w:1280">

# Assuming there are two independent variables. The Greek alphabet is called Beta. The Beta's are the weights (or called Coefficients).
# 
# Taking natural logarithm on both sides:

# <img src="https://img1.wsimg.com/isteam/ip/96331c9c-69bc-4562-9ea2-648231e1d0b3/logistic_3.JPG/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280">

# Taking logarithm on both sides has made the calculation of the output on the right hand side becomes Linear again. The left hand side is also called the "odds" or the probability. Odds is defined as the ratio of the probability of the event happened divided by the probability of the event not happened. For example, if the probability of something happen is 90%, then the probability of not happen is 10%, its odds is 0.9/(1 - 0.9) = 9.
# 
# The weights (or Coefficients) of the model have to be estimated from your data. This is done using maximum-likelihood estimation, which is a statistics terminology. Maximum-Likelihood has been used for a lot of machine learning algorithms.
# 
# To start, we assume that the weights (or coefficients) to be zeros, and after each iteration we refine the weights, in other words, after each iteration we should have a better fit of the model. 
# 
# So much theory for now, let's use some data to illustrate the Logistic Regression process.
# 
# Assume we have a dataset of 10 data points, X1 and X2 :

# <img src="https://isteam.wsimg.com/ip/96331c9c-69bc-4562-9ea2-648231e1d0b3/logistic_dataset.JPG/:/rs=w:1280">

# Where Y is the Actual value of the outcome.
# 
# We start off with assuming that the weights of X1 and X2 are both 0, and the intercept to be 1.
# 
# We performed the calculation in an excel spreadsheet, the output value is calculated using the formula (we can also called the output to be the "predicted" value):

# <img src="https://img1.wsimg.com/isteam/ip/96331c9c-69bc-4562-9ea2-648231e1d0b3/logistic_6-0001.JPG/:/rs=w:1280">

# <img src="https://img1.wsimg.com/isteam/ip/96331c9c-69bc-4562-9ea2-648231e1d0b3/logistic_spreadsheet-0001.JPG/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280">

# We calculate the new coefficients using an update equation:

# <img src="https://isteam.wsimg.com/ip/96331c9c-69bc-4562-9ea2-648231e1d0b3/logistic_5.JPG/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280">

# where b is the coefficient that we want to update. Alpha is what is called Learning Rate in Machine Learning, its value is usually in the range of 0.1 to 0.3, and in this example we choose 0.3 as our learning rate. 
# 
# to calculate the new Beta 0 :    0 + 0.3 x (1 - 0.5) x 0.5 x (1 - 0.5) x 1 = 0.0375
# 
# we can calculate the new Beta 1 and Beta 2 using the same formula. For the first data point, we have come up with the new Beta 1 = 0.259597 and new Beta 2 = 0.066415
# 
# Then we will use the new Coefficients in the next round of iteration.
# 
# We repeat this process and update the model for each data point. In machine learning, we call each iteration an Epoch. When the number of data points are small, as our example, it is recommended we repeat as many as times as possible. In theory, each Epoch should produce a better fit for the model. 
# 
# By the end of each iteration, we can calculate the error value for the model. Let's use the first iteration as an example:

# <img src="https://img1.wsimg.com/isteam/ip/96331c9c-69bc-4562-9ea2-648231e1d0b3/logistic_spreadsheet2.JPG/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280">

# calculated "predicted" output = 0.5, given Y = 1, the Squared Error value is 0.25:

# <img src="https://img1.wsimg.com/isteam/ip/96331c9c-69bc-4562-9ea2-648231e1d0b3/logistic_7.JPG/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280">

# After the iteration of all 10 data points, we have the Root Mean Squared Error (RMSE) for the first Epoch = 0.485278:

# 1. <img src="https://img1.wsimg.com/isteam/ip/96331c9c-69bc-4562-9ea2-648231e1d0b3/logistic_8.JPG/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280">
# 

# If we repeat this process 10 times (Epoch = 10), RMSE will become smaller and smaller, if we plot the RMSE in a graph and you can see that the RMSE is reduced significantly:

# <img src="https://isteam.wsimg.com/ip/96331c9c-69bc-4562-9ea2-648231e1d0b3/logistic_rmse.JPG/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280">

# As you could see in the plot above, after just 1 Epoch the RMSE is 0.485278, and after 10 Epoch the RMSE has reduced to 0.161833
# 
# How do we know how accurate is our model ?
# 
# we measure the Accuracy of the model by comparing the Predicted value vs the Actual value, let's use the first row as an example, the Actual value of Y is 1, our predicted value of Y is 1, therefore the Error is 0. On the other hand, look at the forth row: the Actual value of Y is 0, the predicted value is 1, so the Error is 1.
# 
# We sum up the Errors to calculate the Accuracy : (Total number of Errors / Total number of data points) X 100%
# 
# in the first Epoch, there are 3 Errors, therefore our Accuracy is 70%
# 
# After we repeated 10 Epochs, our Accuracy has achieved 100%, as plotted here in this graph:

# <img src="https://isteam.wsimg.com/ip/96331c9c-69bc-4562-9ea2-648231e1d0b3/logistic_accuracy.JPG/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280">

# In this post, I covered the basic maths behind Logistic Regression. 
# 
# I have uploaded the excel spreadsheet for logistic regression calculation to Github, you can find it here:
# 
# https://github.com/briantfu/logistic_regression

# # Please visit my blog:
# # https://www.ai-intelligence.net
# 
