#!/usr/bin/env python
# coding: utf-8

# This notebook has been run successful in my cmputer and provides an accuracy of 68% on testing data
# I am still learning how to deal with data, analyze it and test it so i am not running for accuracy now.

# In[ ]:


from sklearn import linear_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[ ]:


dataset=pd.read_csv("/kaggle/input/real-estate-price-prediction/Real estate.csv")
dataset.head()


# In[ ]:


dataset.isnull().sum()


# In[ ]:


x=dataset.drop("Y house price of unit area",axis=1)
y=dataset['Y house price of unit area']
y.head()


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=300)
y_test.head()


# In[ ]:


reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
reg.score(x_test,y_test)


# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2)
ax1.scatter(x_train['X2 house age'],y_train,c='r')
ax2.scatter(x_test['X2 house age'],y_test,c='y')
plt.show()


# In[ ]:


y_predict=reg.predict(x_test)
print(y_predict)


# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2)
ax1.scatter(x_test['X2 house age'],y_predict,c='r')
ax2.scatter(x_test['X2 house age'],y_test,c='y')
plt.show()

