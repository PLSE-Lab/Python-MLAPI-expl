#!/usr/bin/env python
# coding: utf-8

# Height of a person depends of factors like genetics,region and nutrition during childhood.In last 150 years it is observed that average height of people has increased.This is due to better nutrition that children receive in developed countries.Here I have done a linear regression on the height of Father and Son.This Kernel shows how to perform Simple linear regression using Python.If you like my work and find it to be useful please do give a vote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Importing python modules

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# # Importing the data 

# In[ ]:


data=pd.read_csv('../input/Father_Son_height_C.csv')


# # Displaying the data

# In[ ]:


data.head()


# In[ ]:


data.info()


# # Getting the Independent variables Fathers height 

# In[ ]:


X=data['Father'].values[:,None]
X.shape


# # Getting the dependent variable son's height'

# In[ ]:


y=data.iloc[:,2].values
y.shape


# # Spliting the data into test and train data

# In[ ]:


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# # Doing a linear regression

# In[ ]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)


# # Predicting the height of Sons

# In[ ]:


y_test=lm.predict(X_test)
print(y_test)


# # Plotting the given data against the predicted data

# In[ ]:


plt.scatter(X,y,color='b')
plt.plot(X_test,y_test,color='black',linewidth=3)
plt.xlabel('Father height in inches')
plt.ylabel('Son height in inches')
plt.show()


# Blue color dots represent the  father son height in the data.The black color line represent the Linear regression line predicted by the alogrithm.

# **Model Performance**

# In[ ]:


y_train_pred=lm.predict(X_train).ravel()
y_test_pred=lm.predict(X_test).ravel()


# In[ ]:


from sklearn.metrics import mean_squared_error as mse,r2_score


# In[ ]:


print("The Mean Squared Error on Train set is:\t{:0.1f}".format(mse(y_train,y_train_pred)))
print("The Mean Squared Error on Test set is:\t{:0.1f}".format(mse(y_test,y_test_pred)))


# The mean squared error value for a good model should have low value.

# In[ ]:


print("The R2 score on the Train set is:\t{:0.1f}".format(r2_score(y_train,y_train_pred)))
print("The R2 score on the Test set is:\t{:0.1f}".format(r2_score(y_test,y_test_pred)))


# The R2 Square error for a good model should be close to 1.
