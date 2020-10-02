#!/usr/bin/env python
# coding: utf-8

# ## **Digit Recognizer with Logistic Regression**

# ### **Content**
# * [Data Exploration](#1)
# * [Normalization](#2)
# * [Train-Test Split](#3)
# * [Initialization of Weights and Bias](#4)
# * [Sigmoid Function For Logistic Regression](#5)
# * [Forward - Backward Propagation](#6)
# * [Updating Parameters Using Forward and Backward Propagation](#7)
# * [Prediction Method For Logistic Regression Model](#8)
# * [Defining The Logistic Regression Method](#9)
# * [Conclusion](#10)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="1"></a> 
# **Data Exploration**

# In[ ]:


trainData = pd.read_csv('../input/train.csv')


# In[ ]:


trainData.head()


# In[ ]:


trainData.info() # We have 42000 samples with 785 columns including label column


# In[ ]:


# Visualization of our digits
sns.countplot(trainData.label)
plt.title('Digit Count', color = 'blue', fontsize = 15)


# * We will use two digits for binary classification. Lets take two label from train data. (3 and 8)

# In[ ]:


trainData = trainData[(trainData["label"] == 3) | (trainData["label"] == 8)] # Filtering the train data


# In[ ]:


trainData.head() # Now we have only 3 and 8 labeled numbers


# In[ ]:


trainData.label = [1 if each == 3 else 0 for each in trainData.label] # We will call 1 for number three and 0 for number eight. We need binary results for logistic regression.


# In[ ]:


y = trainData.label.values # Only labels for model training
x_data = trainData.drop(["label"], axis = 1)


# In[ ]:


x_data.head() # Train data wihtout label.


# <a id="2"></a> 
# **Normalization** <br/>
# 
# &emsp; * Each pixel value is in the range 0..256. We will convert this range to 0..1*

# In[ ]:


x = (x_data / 255.0)


# We have 8414 images. 28x28 pixels consisting of three and eight

# In[ ]:


print('digits({0[0]},{0[1]})'.format(x.shape))


# *Lets look some sample of digits.*

# In[ ]:


def showDigit(index):
    sampleDigit = x.iloc[:,0:].values[index]
    digit = sampleDigit.reshape(28, 28)
    plt.axis('off')
    plt.imshow(digit, cmap = cm.binary) 


# In[ ]:


showDigit(26) # 25 - 26 is random member of our array. Just for check the how the numbers are look


# In[ ]:


showDigit(25)


# <a id="3"></a> 
# ***Train-Test Split***

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[ ]:


x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train : ", x_train.shape)
print("x_test : ", x_test.shape)
print("y_train : ", y_train.shape)
print("y_test : ", y_test.shape)


# <a id="4"></a> 
# **Initialization of weights and bias.**

# In[ ]:


def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w,b


# <a id="5"></a> 
# **Sigmoid function for logistic regression**

# In[ ]:


def sigmoid(z):
    y_head = 1/(1 + np.exp(-z))
    return y_head


# <a id="6"></a> 
# **Forward - Backward Propagation**

# In[ ]:


def forward_backward_propogation(w, b, x_train, y_train):
    
    #forward propogation z = (w.T)x + b 
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
    cost = (np.sum(loss)) / x_train.shape[1]  # x_train.shape[1] is for scaling
    
    #backward propogation
    derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1] # x_train.shape[1] is for scaling
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1] # x_train.shape[1] is for scaling
    gradients = {"derivative_weight" : derivative_weight, "derivative_bias" : derivative_bias}
    
    return cost, gradients


# <a id="7"></a> 
# **Updating Parameters Using Forward and Backward Propagation**

# In[ ]:


def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    costList = []
    costListForPlot = []
    index = []
    
    # updating(learning) parameters is number_of_iteration times
    for i in range(number_of_iteration):
        # make forward and backward propogation and find costs and gradients
        cost,gradients = forward_backward_propogation(w, b, x_train, y_train)
        costList.append(cost)
        #lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 100 == 0:
            costListForPlot.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i, cost))
            
    parameters = {"weight" : w, "bias" : b}
    plt.plot(index, costListForPlot)
    plt.xticks(index, rotation = 'vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, costList


# <a id="8"></a> 
# **Prediction Method for Logistic Regression Model**

# In[ ]:


# After forward and backward propagation. We will predict results from out model.

def predict(w,b, x_test):
    z = sigmoid(np.dot(w.T, x_test) + b)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    
    return Y_prediction


# <a id="9"></a> 
# **Defining The Logistic Regression Method**

# In[ ]:


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    
    # initialize
    dimension = x_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)
    
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations) 
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy : {} %.".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


# * Lets test our model. For 1000 iterations with 0.1 learning rate.

# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 0.1, num_iterations = 1000) 


# <a id="10"></a> 
# ## Conclusion
# 
# * The above transactions are called simple neural network (logistic regression). These are made without sklearn library for learn logic behind logistic regression. <br/>
# 
# Thanks in advance for suggestions.
