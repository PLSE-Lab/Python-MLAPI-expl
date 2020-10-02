#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from  ipywidgets import interact


# importing libraries and modules from which we gonna use the functions to execute Multi-Linear Regression

# In[ ]:


df = pd.read_csv("../input/insurance-1/insurance 1.csv")
df.head()


# uploaded data file of insurance 

# In[ ]:


x = df.iloc[:,0:2].values
y = df.iloc[:,-1].values


# converting our dataset into arrays so that we can use the dataset as input for our functions

# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3, random_state = 16)


# We will make our own split function but for now let's use it from the inbuilt library only
# 

# In[ ]:


model = LinearRegression()
model.fit(xtrain,ytrain)


# fitting the model to the dataset

# In[ ]:


m = model.coef_
c = model.intercept_


# find the value of slope and intercept from our multi regresssion algorithm

# In[ ]:


x = [12, 37]
y_predict = sum(m*x) + c
y_predict


# predicting the value form direct equation

# In[ ]:


x = [12, 37]
y_predict = model.predict([x])
y_predict


# predicting the value from the inbuilt model 

# In[ ]:


ytest_predict = model.predict(xtest)


# calling the function and storing the value to use other time
# 

# In[ ]:


def InsurancePricePredict(Age, bmi):
    y_predict = model.predict([[Age, bmi]])
    print("Insurance Price is:", y_predict[0])


# defining a function to predict the price premium of insurance

# In[ ]:


InsurancePricePredict(33,3)


# In[ ]:


age_min = df.iloc[:, 0].min()
age_max = df.iloc[:, 0].max()
bmi_min = df.iloc[:, 1].min()
bmi_max = df.iloc[:, 1].max()


# defining the limits from min to max of the age and BMI index to design the model

# In[ ]:


interact(InsurancePricePredict, Age = (age_min, age_max), bmi = (bmi_min, bmi_max) )


# using the widget library module we used the interact function to make a model, try and measure your insurance price on the basis of given dataset

# In[ ]:




