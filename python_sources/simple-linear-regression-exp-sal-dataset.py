#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


dataset = pd.read_csv('../input/Salary_Data.csv')
dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


X = pd.DataFrame(dataset, index= range(30), columns=['YearsExperience'])


# In[ ]:


X.info()


# In[ ]:


X.head()


# In[ ]:


y = dataset.loc[: , 'Salary']


# In[ ]:


y.head()


# In[ ]:


plt.scatter(X , y , color = 'yellow')


# **THE ABOVE SCATTER PLOT IS SHOWING A LINEAR RELATIOSHOIP BETWEEN X AND y, SO HERE WE CAN APPLY SIMPLE LINEAR REGRESSION MODEL AS ONLY ONE INDEPENDENT VARIABLE IS THERE **

# In[ ]:


##Splitting the dataset into train test


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (1/3), random_state = 0 )


# In[ ]:


X_train.info(), X_test.info()


# In[ ]:


##Fitting the regression model on the training set


# In[ ]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)


# In[ ]:


##Predicting the y variables


# In[ ]:


y_predict = regressor.predict(X_test)


# In[ ]:


y_predict


# In[ ]:


##Visualising the training dataset


# In[ ]:


plt.scatter(X_train, y_train, color = 'orange')
plt.plot(X_train, regressor.predict(X_train), color = 'pink' )
plt.title('Training Dataset')
plt.xlabel('Experience')
plt.ylabel('Salary')


# In[ ]:


regressor.coef_


# In[ ]:


regressor.intercept_


# In[ ]:


##Visualising the test dataset


# In[ ]:


plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Test Dataset')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:





# In[ ]:




