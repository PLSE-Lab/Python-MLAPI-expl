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


data=pd.read_csv('../input/Bitcoin.csv')


# In[ ]:


data.head()


# In[ ]:


#Here we don't need columns close_ratio, spread, and market. So, i am gonna drop that columns.
data.drop(columns=['close_ratio', 'market', 'spread','slug','symbol','name','ranknow','volume'], inplace = True)
data


# In[ ]:


#As we can see that we have dropped the columns that are not required.
data.shape


# In[ ]:


#We have 2039 rows and 5 columns in the data.


# In[ ]:


#Let's find min, count, max etc now to know more about the data.
data.describe()


# In[ ]:


data.dtypes


# In[ ]:


#As we can see that date is not in the date format. So, i will convert it into date datatype.
data['date'] = pd.to_datetime(data['date'])
data.dtypes


# In[ ]:


#date is successfully converted into date format


# In[ ]:


data.head()


# In[ ]:


#Also we don't need the columns high and low. So, i am gonna drop them too.
data.drop(columns=['high', 'low'], inplace = True)
data.head()


# In[ ]:


#linear regression
from sklearn.linear_model import LinearRegression


# In[ ]:


#splitting the data into train and test set
from sklearn.model_selection import train_test_split


# In[ ]:


#Let Set the date as index
data.set_index('date', inplace = True)


# In[ ]:


#import the matplot library
from matplotlib import pyplot as plt


# In[ ]:


x = data['open']
y = data['close']
plt.figure(figsize=(15,12))
plt.plot(x, color='red')
plt.plot(y, color = 'blue')
plt.show()
plt.xlabel('Open')
plt.ylabel('Close')
#plotting the graph between open and close attribute to see the relation between them


# In[ ]:


#Sorting the index 
data.sort_index(inplace=True)


# In[ ]:


X = data['open']
Y = data['close']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
X_test.tail()
#random_state-> same random splits


# In[ ]:


reg=LinearRegression()


# In[ ]:


x_train = np.array(X_train).reshape(-1,1)
x_test = np.array(X_test).reshape(-1,1)


# In[ ]:


reg.fit(x_train,y_train) 


# In[ ]:


reg.score(x_train, y_train)


# In[ ]:


plt.figure(figsize=(15,12))
plt.plot(y_train)
plt.plot(y_test)

