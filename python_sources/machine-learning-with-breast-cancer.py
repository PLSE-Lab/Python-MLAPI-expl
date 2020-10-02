#!/usr/bin/env python
# coding: utf-8

#  ## Breast Cancer
# 

# <a href="https://ibb.co/f2WVf3d"><img src="https://i.ibb.co/TkGD6QM/images.jpg" alt="images" border="0"></a>

# <font color='red'> <br>
# 
# * [A. Problem Understanding](#1)
#     * [Data Describtion](#2)
#     * [Exploratory Data Analysis(EDA)](#3)    
# * [B. Logistic Regression](#4)
#     * [Creating Parameters](#5)
#     * [Forward and Backward Propagation](#6)
#     * [Implementing Update Parameters](#7)
#     * [Prediction Parameter](#8)
# * [C. Logistec Regression with Sklearn](#9)
#    
#   [CONCLUSION](#10)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id="1"></a> <br>
# ## A. Problem Understanding
# 
# Despite a great deal of public awareness and scientific research, breast cancer continues to be the most common cancer and the second largest cause of cancer deaths among women. Approximately 12% of U.S. women will be diagnosed with breast cancer, and 3.5% will die of it. The annual mortality rate of approximately 28 deaths per 100,000 women has remained nearly constant over the past 20 years. 

# <a id="2"></a> <br>
# ## Data Describtion

# *  ID number
# *  Diagnosis (M = malignant, B = benign)
# *  Ten real-valued features are computed for each cell nucleus:
# *  radius (mean of distances from center to points on the perimeter)
# *  texture (standard deviation of gray-scale values)
# *  perimeter
# *  area
# *  smoothness (local variation in radius lengths)
# *  compactness (perimeter^2 / area - 1.0)
# *  concavity (severity of concave portions of the contour)
# *  concave points (number of concave portions of the contour)
# *  symmetry
# *  fractal dimension ("coastline approximation" - 1)
#  Note: Mean, Etandard Error (SE) and Worst (mean of the three largest values) of these features are obtained from each image, resulting in 30 features. For example, the third column is Mean Radius, column 13 is Radius SE, column 23 is Worst Radius. All feature values are stored with four significant numbers.

# In[ ]:


data=pd.read_csv('../input/breast-cancer/breast-cancer.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


data.drop(['id','Unnamed: 32'],axis=1,inplace=True)


# In[ ]:


data.head()


# <a id="3"></a> <br>
# Exploratory Data Analysis(EDA)

# In[ ]:


data.describe()


# In[ ]:


data2= data[['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean',
                 'smoothness_mean','area_worst','concavity_mean','concave points_mean',
                 'symmetry_mean','fractal_dimension_mean']]


# In[ ]:


color_list = ['cyan' if i=='M' else 'lime' for i in data2.loc[:,'diagnosis']]
pd.plotting.scatter_matrix(data2.loc[:, data2.columns != 'diagnosis'],
                           c=color_list,
                           figsize= [20,20],
                           diagonal='hist',
                           alpha=0.5,
                           s = 200,
                           marker = '*',
                           edgecolor= "black")
                                        
plt.show()


# In[ ]:


# Values of 'Benign' and 'Malignant' cancer cells
data.diagnosis.value_counts()


# In[ ]:


#Visualization
sns.countplot(x="diagnosis", data=data)
plt.show()


# In[ ]:


g = sns.jointplot(data.radius_mean, data.smoothness_mean, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()


# In[ ]:


plt.figure(figsize=(25,25))
sns.heatmap(data.corr(),annot=True,cmap='RdBu_r')


# <a id="4"></a> <br>
# ## B. Logistic Regression

# <a href="http://ibb.co/c574qx"><img src="http://preview.ibb.co/cxP63H/5.jpg" alt="5" border="0">

# In[ ]:


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
print(data.info())
#Firstly M and B values update 0 and zero. Because not using string.


# In[ ]:


y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)


# In[ ]:


#normalization
#This is a formul>>   (x - min(x))/(max(x)-min(x))
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values


# In[ ]:


x.shape
# change value (matrix)


# In[ ]:


# %30 testing %70 training ///  random constant=42
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


# <a id="5"></a> <br>
# ## Creating Parameters

# * Parameters are weight and bias.
# * Weights: coefficients of each pixels
# * Bias: intercept
# * z = (w.t)x + b => z equals to (transpose of weights times input x) + bias
# * y_head = sigmoid(z)
# * Sigmoid function makes z between zero and one so that is probability.

# In[ ]:


# dimension =feauture values

def initialize_weights_and_bias(dimension):
    
    w = np.full((dimension,1),0.01)
    b = 0.0  #float values
    return w,b


# In[ ]:


look = np.full((6,1),0.01)
print(look)


# Sigmoid formule and graphic
# 
# <a href="https://ibb.co/Njcnpqj"><img src="https://i.ibb.co/jMnZGYM/images.png" alt="images" border="0"></a>
# 
# 

# In[ ]:


# Let's calculating z
# z = np.dot(w.T,x_train)+b

def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    return y_head


# In[ ]:


print(sigmoid(-6))
print(sigmoid(0))
print(sigmoid(6))


# <a id="6"></a> <br>
# ## Forward and Backward Propagation

# Now if our cost is going to be a mistake. we must create backward and forward propagation.

# Loss formule
# <a href="https://imgbb.com/"><img src="https://image.ibb.co/eC0JCK/duzeltme.jpg" alt="duzeltme" border="0"></a>

# Cost
# <a href="http://imgbb.com/"><img src="http://image.ibb.co/dAaYJH/7.jpg" alt="7" border="0"></a>
# 

# In[ ]:


x_train.shape[1] #for scaling


# In[ ]:


def forward_backward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]  
    
    # backward propagation
    #weight turev
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] 
    #bias turev
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]           
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients


# <a id="7"></a> <br>
# ## Implementing Update Parameters

# In[ ]:


def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


#  <a id="8"></a> <br>
# ## Prediction Parameter

# In[ ]:


def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction


# In[ ]:


x_test.shape[1]


# In[ ]:


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 3, num_iterations = 300)


# <a id="9"></a> <br>
# ## C.Logistec Regression with Sklearn
# 
# With the Sklearn library, we can find the result you found above in a much easier way.

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))


# <a id="10"></a> <br>
# 
# ##  CONCLUSION

# Thank you for your votes and comments
# 
# If you have any suggest, May you write for me, I will be happy to hear it.
