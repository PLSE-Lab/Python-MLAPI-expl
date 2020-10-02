#!/usr/bin/env python
# coding: utf-8

# # Please do Vote up if you liked my work

# Context
# 
# Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.
# 
# He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.
# 
# Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.
# 
# In this problem you do not have to predict actual price but a price range indicating how high the price is

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


# **Load Data and libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split


# In[ ]:


data=pd.read_csv('../input/mobile-price-classification/train.csv')
test=pd.read_csv('../input/mobile-price-classification/test.csv')


# # Work With Data

# **Data Visualization **

# In[ ]:


fig = plt.figure(figsize = (15,20))
ax = fig.gca()
data.hist(ax = ax)


# **Check the shape of the data**

# In[ ]:


print(data.shape)
data.head()


# In[ ]:


print(test.shape)
test.head()


# **Data Columns Name**

# In[ ]:


data.columns


# **Are There any Null Values??**

# In[ ]:


data.isnull().sum()


# **Data Types **
# No String Data

# In[ ]:


data.dtypes


# # Spliting Data To X and Y

# In[ ]:


X=data.drop(['price_range'],axis=1)


# Our Target is price_range

# In[ ]:


Y=data['price_range']


# Spliting data to use it in the model

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=4,test_size=0.2)


# # Bulding Our Model 
# We will use LinearRegression model

# In[ ]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# # Check How the model is good ?
# 

# In[ ]:


from sklearn.metrics import mean_squared_error
predict=lr.predict(x_test)
mean_squared_error(predict,y_test)


# In[ ]:


lr.score(x_test,y_test)


# In[ ]:


plt.scatter(y_test,predict)
plt.show()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)


# In[ ]:


knn_predict=knn.predict(x_test)
print(knn.score(x_test,y_test))
print(mean_squared_error(knn_predict,y_test))


# # Conclusion: KNN performed the best

# # Now Let's Predict the price Range of the test file

# In[ ]:


test=test.drop(['id'],axis=1)
test_predict=knn.predict(test)


# In[ ]:


print(test_predict)


# In[ ]:


pd.DataFrame(test_predict).head()


# In[ ]:


test['price_range']=test_predict
test.head(50)


# In[ ]:




