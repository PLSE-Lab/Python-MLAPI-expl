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


#import the required librariers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split#for spliting the data into training and testing
from sklearn.metrics import accuracy_score,r2_score
from ipywidgets import interact #to display various input values

#attaching the data set 
df=pd.read_csv("../input/second-hand-used-cars-data-set-linear-regression/train.csv")
df.head()
df.shape
#converting the independent variable into 2D and dependent variable into 1D
x=df.iloc[:,1:4].values
y=df.iloc[:,-1].values


# In[ ]:


#creating with predefined 
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=13)
print(xtrain[0][0])


# In[ ]:


#Creating with our own functions
def Train_Test_Split(x,y,test_size=0.5,random_state=None):
    n=len(y)
    if len(x)==len(y):
        if random_state:
            np.random.seed(random_state)
        shuffle_index=np.random.permutation(n)
        x=x[shuffle_index]
        y=y[shuffle_index]
        test_data=round(n*test_size)
        xtrain,xtest=x[test_data:],x[:test_data]
        ytrain,ytest=y[test_data:],y[:test_data]
        return xtrain,xtest,ytrain,ytest
    else:
        print("Data should be in same size.")
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=11)
print(xtrain[0][0])


# In[ ]:



#training the machine
model=LinearRegression()
model.fit(xtrain,ytrain)


# In[ ]:


m=model.coef_
c=model.intercept_


# In[ ]:


#prediction by mathematics by using the formula
#y=m1*x1+m2*x2+m3*x3+c
x=[3,2,1]
y_predict=(sum(m*x)+c)
y_predict


# In[ ]:


#prediction of carprice
ytest_predict=model.predict(xtest)


# In[ ]:


def CarPricePredict(kilometer,condition,years):
    y_predict=model.predict([[kilometer,condition,years]])
    print("CarPrice is :",y_predict[0])


# In[ ]:


CarPricePredict(4,12,33)


# In[ ]:


kil_min=df.iloc[:,1].min()
kil_max=df.iloc[:,1].max()
cond_min=df.iloc[:,2].min()
cond_max=df.iloc[:,2].max()
yrs_min=df.iloc[:,3].min()
yrs_max=df.iloc[:,3].max()


# In[ ]:


interact(CarPricePredict, kilometer  = (kil_min, kil_max),condition = (cond_min, cond_max), years = (yrs_min, yrs_max))

