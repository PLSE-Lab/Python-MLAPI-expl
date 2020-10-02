#!/usr/bin/env python
# coding: utf-8

# <h1> Problem Statement: Prediction of weight of new born baby</h1>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# set the working directory
os.chdir("../input")


# In[ ]:


# get the data
df=pd.read_csv("birthwt.csv")


# In[ ]:


df.head(10)


# * Here target variable "bwt"(birth weight) is continuous in nature.
# * So prediction is done by regression method
# 

# In[ ]:


df.shape


# In[ ]:


# divide the data set in two parts : train and test
train,test=train_test_split(df,test_size=0.2)


# In[ ]:


train.shape, test.shape


# ## Decision Trees

# In[ ]:


#train the model
fit_DT=DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:9],train.iloc[:,9])
fit_DT


# In[ ]:


# apply model on the test data
predictions_DT=fit_DT.predict(test.iloc[:,0:9])
predictions_DT


# In[ ]:


#model evaluation
#calculate the MAPE 
def MAPE(y_true,y_pred):
    mape=np.mean(np.abs((y_true-y_pred)/y_true))*100
    return mape


# In[ ]:


MAPE(test.iloc[:,9],predictions_DT)


# ## Linear Regression

# In[ ]:


# import required library and module
import statsmodels.api as sm


# In[ ]:


# train the model using training dataset
model= sm.OLS(train.iloc[:,9],train.iloc[:,0:9]).fit()


# In[ ]:


# get the summary of the model
model.summary()


# In[ ]:


# make predictions using the above models
predictions_LR=model.predict(test.iloc[:,0:9])


# In[ ]:


# model evaluation
MAPE(test.iloc[:,9],predictions_LR)


# In[ ]:




