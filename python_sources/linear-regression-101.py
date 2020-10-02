#!/usr/bin/env python
# coding: utf-8

# <b>Linear Regression</b><br>

# In[3]:


import numpy as np                         
import pandas as pd      
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


# <b>Load Dataset</b><br>

# In[6]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# <b>Data Visualization</b><br>

# In[7]:


train.plot(x = "x", y = "y", style = "o")
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")


# <b>Split data into input and output feature</b><br>

# In[8]:


train_x = train.iloc[:, :-1].values
train_y = train.iloc[:, :1].values
test_x = test.iloc[:, :-1].values
test_y = test.iloc[:, :1].values


# <b>Model Preparation and Fitting</b><br>

# In[9]:


model = LinearRegression()                           #Creating linear regression object
model.fit(train_x, train_y)                          #Fitting model to our training dataset
prediction = model.predict(test_x)                   #Predicting the values of test data


# <b>Metrices</b><br>

# In[10]:


print("The coefficient is: ", model.coef_)
print("The mean squared error is {0:0.2f}".format(mean_squared_error(test_y, prediction)))
print("Variance score is {0:0.2f}".format(r2_score(test_y, prediction)))


# <b>Final Plots</b>

# In[11]:


plt.scatter(test_x, test_y, color = 'black')
plt.plot(test_x, prediction, linewidth = 3)
plt.show()


# <b>Final Data Frame</b>

# In[12]:


data_test_prediction = pd.DataFrame({'Test_X': test_x[:,0], 'Test_Y': test_y[:,0], 'Predictions': prediction[:,0]})
data_test_prediction.head(n = 10)

