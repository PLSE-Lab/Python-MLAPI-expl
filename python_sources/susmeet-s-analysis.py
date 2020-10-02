#!/usr/bin/env python
# coding: utf-8

# making all the required imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing


# Let us import the data into a pandas DataFrame

# In[ ]:


history = pd.read_csv('../input/Historical Product Demand.csv')


# **EDA**
# Before moving on to any kind of analysis,
# Let us explore our data, it's size and columns

# In[ ]:


history[:5]


# Each row has got the Order_Demand at a Warehouse for a particular product on a particluar date.

# In[ ]:


history.shape


# In[ ]:


len(history['Product_Category'].unique())


# In[ ]:


len(history['Product_Code'].unique())


# So in our historical data, we have 2160 different products across 33 different categories

# In[ ]:


dates = [pd.to_datetime(date) for date in history['Date']]
dates.sort()


# In[ ]:


dates[0]


# In[ ]:


dates[-1]


# The dates in our data range from 8th January '11 to 9th January  '17

# **Problem Statement**
# We need to be able to predict the demand in the upcoming days.
# Let us see if we could find some correlation between the month, day and the demand

# **Preparing DataSets**
# One more thing to notice here is, all of our features are categorical and none are continuous, we shall use a decision tree in such a case

# In[ ]:


X = history[['Product_Code','Warehouse','Product_Category']]
Y = history[['Order_Demand']]


# In[ ]:


from sklearn import preprocessing
def encode(x):
    le = preprocessing.LabelEncoder()
    return le.fit_transform(x)


# In[ ]:


for column in X.columns:
    X[column] = encode(X[column])


# In[ ]:


from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)


# In[ ]:




