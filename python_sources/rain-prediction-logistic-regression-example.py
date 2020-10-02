#!/usr/bin/env python
# coding: utf-8

# **IS IT WHETHER OR NOT GOING TO RAIN IN AUSTRALIA?**
# 
# In this kernel I use rain in Australia dataset. This dataset contains daily weather observations from numerous Australian weather stations.
# 
# The target variable RainTomorrow means: Did it rain the next day? Yes or No.
# 
# I used **logistic regression model** for predictions. Before using sklearn library for logistic regression, I explained all the steps with the help of DATAI Team's wonderful kernel (https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners) and Udemy course (https://www.udemy.com/machine-learning-ve-python-adan-zye-makine-ogrenmesi-4/).
# 
# In order to create a logistic regression model, I followed these steps:
# * Importing libraries
# * Importing dataset
# * Cleaning dataset
# * Converting Predictions to Binary form (0 or 1)
# * Excluding Tomorrow's Prediction from the Dataset (Because I wouldn't want my model to cheat)
# * Normalization Progress (Scaling all the values between 0 and 1)
# * Splitting Dataset for Training and Testing
# * Creating the Initial Parameters (Weight and Bias)
# * Defining the Sigmoid Function
# * Defining Forward and Backward Propagation
# * Defining Update Parameters Method
# * Defining Prediction Method
# * Implementing Logistic Regression Using Test Data
# 
# And at the end of this kernel I showed an easier way of doing logistic regression which is using sklearn library.

# **1. IMPORTING LIBRARIES**

# In[105]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **2. DATA PREPARATION**

# **2.1. Importing Data**

# In[106]:


data = pd.read_csv('../input/weatherAUS.csv')
data.sample(5)


# **2.2. Cleaning Data**

# In[107]:


# Getting rid of the columns with objects which will not be used in our model:
data.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RISK_MM'], axis=1, inplace=True)
data.head(5)


# In[108]:


# And we need to replace NaN values with mean values of each column:
data.fillna(data.mean(), inplace=True)
data.head(5)


# **2.3. Converting Predictions to Binary for Logistic Regression**

# In[109]:


# Now we can change that day and next days'predictions (yes and no) to 1 and 0:
data.RainToday = [1 if each == 'Yes' else 0 for each in data.RainToday]
data.RainTomorrow = [1 if each == 'Yes' else 0 for each in data.RainTomorrow]
data.sample(3)


# **2.4. Excluding Tomorrow's Prediction from the Dataset**

# In[110]:


y = data.RainTomorrow.values
x_data = data.drop('RainTomorrow', axis=1)
x_data.head()


# **2.5. Normalization Progress**

# In order to be able to analyse and visualize all feature values we need to scale all the values between 0 and 1. This progress is called normalisation.
# 
# Normalization formula = (x - min(x)) / (max(x) - min(x))

# In[111]:


# In order to scale all the features between 0 and 1:
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
x.head(5)


# **2.6. Dividing Dataset for Training and Testing the Model**

# 80% of the dataset will be used for training the model and 20% will be used later for testing our model's accuracy.

# In[112]:


# importing sklearn's library for splitting our dataset:
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=75)

# For our matrix calculations we need to transpose our matrixis:
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

print('x_train shape is: ', x_train.shape)
print('y_train shape is: ', y_train.shape)
print('x_test shape is: ', x_test.shape)
print('y_test shape is: ', y_test.shape)


# **3. LOGISTIC REGRESSION**

# **3.1. Creating the Initial Parameters (Weight and Bias)**

# In[113]:


def initialize_weight_bias(dimension):
    w = np.full((dimension,1), 0.01)    # Create a matrix by the size of (dimension,1) and fill it with the values of 0.01
    b = 0.0
    return w,b


# **3.2. Defining the Sigmoid Function**

# Sigmoid Function is used to scale z values between 0 and 1. But this is not the same thing as the normalization.
# 
# Sigmoid function is used for explaining propability.
# 
# Let's say we have used our weight, bias and x values in this formula: **z = w*x + b**
# 
# And let's say our result for **z=5**. And when we apply it to sigmoid function we have **y_head=0.8**
# 
# **That means our model's prediction's result is 1. Because all y_head values above 0.5 (treshold value) on the graph are 1 in the sigmoid function graph.** If we have y_head = 0.4 that means our model's prediction is 0. 
# 
# Here is the sigmoid function's graph:
# ![](https://imgur.com/z1hx3dc.png)

# In[114]:


def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head


# **3.3. Defining Forward and Backward Propagation**

# Loss function tells us if our model's prediction is correct or wrong. If the value of loss function is 1 that means our model's prediction is wrong.
# 
# Loss (error) function formula = -(1 - y) * log(1 - y_head) - y * log(y_head)
# 
# Cost function is the summation of all the loss functions. If cost function is high that means our model makes more mistakes in it's predictions.

# In[115]:


def forward_backward_propagation(w, b, x_train, y_train):
    # forward propagation:
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    
    loss = -(1 - y_train) * np.log(1 - y_head) - y_train * np.log(y_head)     # loss function formula
    cost = (np.sum(loss)) / x_train.shape[1]                               # cost function formula
    
    # backward propagation:
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    
    gradients = {'derivative_weight': derivative_weight, 'derivative_bias': derivative_bias}
    
    return cost, gradients


# **3.4. Defining Update Parameters Method**

# We are going to tell our model how many times to initiate forward and backward propagation.
# 
# For this purpose we will use weight, bias, training datasets (x_train, y_train), number of iteration (how many times to go for-back) and our model's learning rate (how many steps our model uses while doing for-back propagation).

# In[116]:


def update(w, b, x_train, y_train, learning_rate, nu_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    # Initialize for-back propagation for the number of iteration times. Then updating w and b values and writing the cost values to a list:  
    for i in range(nu_of_iteration):
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
    
        # Update weight and bias values:
        w = w - learning_rate * gradients['derivative_weight']
        b = b - learning_rate * gradients['derivative_bias']
        # Show every 20th value of cost:
        if i % 20 == 0:
            cost_list2.append(cost)
            index.append(i)
            print('Cost after iteration %i: %f' %(i,cost))
    
    parameters = {'weight': w, 'bias':b}
    
    # Visulization of cost values:
    plt.plot(index, cost_list2)
    plt.xlabel('Nu of Iteration')
    plt.ylabel('Cost Function Value')
    plt.show()
    
    return parameters, gradients, cost_list


# **3.5. Defining Prediction Method**

# Now we can define our prediction method. For our model's training we have used training data but for prediction we will be using our test data.
# 
# 

# In[117]:


def prediction(w, b, x_test):
    z = sigmoid(np.dot(w.T, x_test) + b)
    y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
            
    return y_prediction


# **3.6. Implementing Logistic Regression Using Test Data**

# Now it's time for using our x_test data and defined methods to predict y_prediction values. 
# 
# Then we will use our predictions (y_test_predictions) and real y values (y_test) to compare and calculate our model's accuracy.

# In[118]:


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, nu_of_iteration):
    dimension = x_train.shape[0]
    w, b = initialize_weight_bias(dimension)    # Creating an initial weight matrix of (x_train data[0] x 1)
    
    # Updating our w and b by using update method. 
    # Update method contains our forward and backward propagation.
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, nu_of_iteration)
    
    # Lets use x_test for predicting y:
    y_test_predictions = prediction(parameters['weight'], parameters['bias'], x_test) 
    
    # Investigate the accuracy:
    print('Test accuracy: {}%'.format(100 - np.mean(np.abs(y_test_predictions - y_test))*100))


# Let's test our model. Use logistic_regression method, train and test data, with a learning rate of 1 and run the forward backward propagation 400 times to train our model. And then show us our cost function (which should decline by the number of iterations) and accuracy of the model.

# In[119]:


logistic_regression(x_train, y_train, x_test, y_test, learning_rate=1, nu_of_iteration=400)


# **4. LOGISTIC REGRESSION WITH SKLEARN LIBRARY**

# In[120]:


# Importing sklearn library for logistic regression:
from sklearn.linear_model import LogisticRegression

# Creating our model named 'lr'
lr = LogisticRegression()

# Training it by using our train data:
lr.fit(x_train.T, y_train.T)

# Printing our accuracy by using our trained model and test data:
print('Test accuracy of sklearn logistic regression library: {}'.format(lr.score(x_test.T, y_test.T)))


# **5. CONCLUSION**
# 
# Logistic regression is a very important subject for understanding machine learning and first step to deep learning.
# That's why it's so important to understand logic behind the logistic regression before using library.
# 
# Key steps for logistic regression are:
# 
#         Clean and fill the data
#         
#             Seperate x and y values
#         
#                 Normalize values
#             
#                     Split train and test data
#                 
#                         Define the initial weight and bias values
#                     
#                             Apply sigmoid function
#                         
#                                 Implement forward and backward propagations
#                             
#                                     Update w and b 
#                                 
#                                         Predict the y test values by using x test and updated w and b
#                                     
#                                             Compare predicted y values and y test values for accuracy
# 
# 
# I hope you understand and liked my logistic regression kernel. If you liked it please vote it and comment your questions or ideas.
# 
# Melih..
