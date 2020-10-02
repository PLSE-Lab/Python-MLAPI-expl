#!/usr/bin/env python
# coding: utf-8

# In[4]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[5]:


#Read data
house = pd.read_csv('../input/housing.csv')
house.head()


# In[6]:


#Size of the datahouse.shape
house.corr()


# In[7]:


corr = house.corr()
corr.sort_values(["median_house_value"], ascending = False, inplace = True)
print(corr.median_house_value)


# **Take 3* polynomials of features where corelation wth median_house_value greater than 0.5**

# * house['median_income-s2'] = house['median_income']**2
# * house['median_income-s3'] = house['median_income']**3
# * house['median_income-sq'] = np.sqrt(house['median_income']
# 
# 
# 

# In[8]:


#How house data looking
house.info()


# In[9]:


#Check for missing values
house.isnull().any()
#Missing data in total_bedrooms


# In[10]:


#Handling missing values
median = house['total_bedrooms'].median()
#median=0
house['total_bedrooms'].fillna(median,inplace=True)
house.isnull().any()


# **Handling Text and Categorical attributes**

# In[11]:


house_cat = house['ocean_proximity']
house_cat.head(10)


# In[12]:


house_cat.value_counts()


# In[13]:


y = house['median_house_value']
house = house.drop(['median_house_value'],axis=1)


# In[14]:


house.head()


# In[15]:


house['ocean_proximity'] = house['ocean_proximity'].astype('category')
house["ocean_proximity1"] = house["ocean_proximity"].cat.codes
house.head()


# In[16]:


#cat_data = house['ocean_proximity']
house = house.drop(['ocean_proximity'],axis = 1)
house.head()


# **Feature Scaling**

# Means subtracts mean value and divide it by variance

# In[17]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(house)
housing_data = scaler.transform(house)


# In[18]:


full_data = housing_data
print(full_data.shape)


# **Spliting Dataset**

# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(full_data, y, random_state=1)


# In[20]:


# default split is 75% for training and 25% for testing
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# **Linear Regression:
# **

# In[21]:


# import model
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)


# In[22]:


# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)


# In[23]:


# make predictions on the testing set
y_pred = linreg.predict(X_test)
print(y_pred)


# In[24]:


import numpy as np
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# **Closed Form**

# In[25]:


import numpy as np
z = np.zeros((X_train.shape[0],1))+1
X_train = np.append(z, X_train, axis=1)
z = np.zeros((X_test.shape[0],1))+1
X_test = np.append(z, X_test, axis=1)


# In[26]:


import numpy

def get_best(X, y):  
    #best_data=numpy.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
    best_data = numpy.matmul(numpy.linalg.inv(numpy.matmul(X.T,X)),numpy.matmul(X.T,y))
    return best_data # returns a list 


# In[27]:


res=get_best(X_train, y_train)
#print(X)
print(res)
res.shape
X_test.shape


# In[28]:


# make predictions on the testing set


y_pred = X_test.dot(res)
print(y_pred)


# In[29]:


import numpy as np
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# **Gradient Descent**

# In[30]:


y_train=np.reshape(y_train, (y_train.shape[0], 1))
y_test=np.reshape(y_test, (y_test.shape[0], 1))
X_train.shape


# In[31]:


def GradientDescent(X, y, rate, theta,itra):
    m = y.shape[0]
    for i in range(itra):
       # print(y_train)
        error =(numpy.matmul(X,(theta.T))-y)
      #  print(error)
        gradient = numpy.matmul(X.T,(error))
        
        theta = theta - (rate/m) *(gradient.T)
    
    return theta


# In[32]:


column=X_train.shape[1]
theta = np.matrix([0 for x in range(column)])
training_rate = 0.46
itra = 1000
res = GradientDescent(X_train, y_train,training_rate,theta,itra)
res


# In[33]:


y_pred = X_test.dot(res.T)
import numpy as np
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# **Linear Regression Newton form**

# In[34]:


def Newton_GradientDescent(X, y, hessian, theta,itra):
    m = y.shape[0]
    for i in range(itra):
       # print(y_train)
        error =(numpy.matmul(X,(theta.T))-y)
      #  print(error)
        gradient = numpy.matmul(X.T,(error))
        
        theta = theta - (1/m)*(np.matmul(gradient.T,hessian))
    
    return theta


# In[35]:


hessian = np.linalg.inv((1/X_train.shape[0])*np.matmul(X_train.T,X_train))
#hessian = (1/X_train.shape[0])*hessian
column=X_train.shape[1]
theta = np.matrix([0 for x in range(column)])
itra = 10
res = Newton_GradientDescent(X_train, y_train,hessian,theta,itra)
res


# In[36]:


y_pred = X_test.dot(res.T)
import numpy as np
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# **Linear Regression Ridge form**

# In[37]:


def Ridge(X, y, rate, theta,itra,lamb):
    m = y.shape[0]
    for i in range(itra):
        error =(numpy.matmul(X,(theta.T))-y)
        grad = numpy.matmul(X.T,error)
        reg = lamb*theta
        theta = theta - ((rate/m)*(grad.T)+reg)
    return theta


# In[38]:


column=X_train.shape[1]
theta = np.matrix([0 for x in range(column)])
training_rate = 0.46
itra = 1000
lamb = 0.001
res = Ridge(X_train, y_train,training_rate,theta,itra,lamb)
res


# In[39]:


y_pred = X_test.dot(res.T)
import numpy as np
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# **Linear Regression Lasso form**

# In[ ]:


from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
import numpy as np
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

