#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


# Load the data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[3]:


train.columns


# In[4]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()

feature_column_names = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

predicted_class_name = ['SalePrice']

X = train[feature_column_names].values
y = train[predicted_class_name].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model.fit(X_train, y_train)


# In[5]:


model.score(X_test, y_test)


# In[6]:


new = test[feature_column_names].values

predicted_prices = model.predict(new)

predicted_price = np.reshape(predicted_prices, -1)

print(predicted_price)


# In[7]:


submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_price})

submission.to_csv("houseprices.csv",index=False)


# In[ ]:




