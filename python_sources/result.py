#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from pandas import read_csv, DataFrame, Series


# In[ ]:


data = read_csv('../input/data.csv')
data.head()


# In[ ]:


data.count(axis=0)


# In[ ]:


df = data
data_train = df.dropna()
df = data
data_test = df.loc[df['admit'] != df['admit']]
print(data_train)
print(data_test)


# In[ ]:


Y_train = data_train.admit
X_train = data_train.drop(['admit'], axis=1)

Y_test = data_test.admit
X_test = data_test.drop(['admit'], axis=1)


# In[ ]:


lr = LogisticRegression(penalty='l1', tol=0.01)
model = lr.fit(X_train, Y_train)


# In[ ]:


y = model.predict(X_train)


# In[ ]:


y = model.predict_proba(X_test)
count = 0

for i in range(len(y)):
    if y[i][1] > 0.4:
        count += 1
        
print(count)

