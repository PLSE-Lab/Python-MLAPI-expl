#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[ ]:


salary=pd.read_csv("../input/salary-data-simple-linear-regression/Salary_Data.csv")


# Checking the data for data type, null values, correlation, distribution etc

# In[ ]:


salary.head()


# In[ ]:


salary.dtypes


# In[ ]:


salary.isnull().sum()


# In[ ]:


salary.corr()


# In[ ]:


salary.info()


# In[ ]:


salary.describe()


# In[ ]:


sns.regplot(x="YearsExperience",y="Salary",data=salary)
plt.show()


# Dividing the data into train and test

# In[ ]:


X=salary.iloc[:,:-1].values
y=salary.iloc[:,1:2].values


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)


# In[ ]:


#Creating and fitting model
linereg=LinearRegression()


# In[ ]:


linereg.fit(X_train,y_train)


# Equation coefficient and Y-intercept

# In[ ]:


linereg.coef_


# In[ ]:


linereg.intercept_


# In[ ]:


# Prediction
y_predict=linereg.predict(X_test)


# In[ ]:


y_predict


# In[ ]:


# Lets see what was y_test
y_test


# We can see the values in y_pred and y_test are nearly similar.
# Lets check for linearityand model accuracy with graph and RMSE,MAE, R-squared values.

# In[ ]:


plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,linereg.predict(X_train),color='green')
plt.title("Years Experience Vs Salary(Train Data)")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()


# In[ ]:


plt.scatter(X_test,y_predict,color='red')
plt.plot(X_train,linereg.predict(X_train),color='blue')
plt.title('Years Experience Vs Salary (Test Data)')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:


print('Mean Abs Error:          ',metrics.mean_absolute_error(y_test,y_predict))
print('Mean Squared Error:      ',metrics.mean_squared_error(y_test,y_predict))
print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
print('R squared value:         ',metrics.r2_score(y_test,y_predict))


# Value of Rsquared(0.9881) shows our model is 98.81% accurate.

# In[ ]:




