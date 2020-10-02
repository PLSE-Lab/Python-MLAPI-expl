#!/usr/bin/env python
# coding: utf-8

# This is an attempt to use <b>Linear Regression</b> for prediction of Housing prices. The dataset used here is Kaggle's <b>Problem on Housing Dataset</b> by <b>Ananth Reddy</b>
# 

# In[27]:


import pandas as pd


# In[28]:


df=pd.read_csv("../input/Housing.csv")


# I tried to use dummies here for the categorical/textual features of the dataset

# In[29]:


dummies=pd.get_dummies(df)


# We know that there must be (k-1) dummies for k variables in order to escape from <b>dummy variable trap</b>. So now I am dropping or deleting the access ones and also to classify two classes we need only one variable.

# In[30]:


dummies=dummies.drop(["driveway_no","recroom_no","fullbase_no","gashw_no","airco_no","prefarea_no"],axis=1)


# In[31]:


y=dummies.price
df_dummies=dummies
x=dummies.drop(["price"],axis=1)


# Train-Test split

# In[32]:


from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)


# Start of our <b>Linear Model</b>

# In[33]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xTrain,yTrain)


# <b>Testing the model</b>

# In[34]:


y_predict=model.predict(xTest)


# <b>Simple accuracy</b> 

# In[35]:


model.score(xTest,yTest)


# <b>RMSE</b> and <b>MSE</b> calculations

# In[36]:


from sklearn.metrics import mean_squared_error 
MSE=mean_squared_error(yTest,y_predict)

from math import sqrt
RMSE=sqrt(MSE)
RMSE


# So, The **RMSE** value can be reduced by having a **larger dataset** and also by **using cross-validations.** 

# In[ ]:




