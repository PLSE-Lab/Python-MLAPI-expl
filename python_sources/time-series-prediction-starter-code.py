#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression


# In[3]:


train = pd.read_csv("../input/ts_train.csv")
test = pd.read_csv("../input/ts_test.csv")


# In[4]:


def select_ts(index):
    return (train[train.tsID == index].copy(),
            test[test.tsID == index].copy())


# In[5]:


D = 30


# In[6]:


def prepare(data):
    train_matrix = []
    test_vector = []
    data  = data.ACTUAL.values
    for i in range(D, len(data)):
        train_matrix.append(data[i-D:i])
        test_vector.append(data[i])
    return np.array(train_matrix), np.array(test_vector)


# In[7]:


def main(train, test):
    M, Y = prepare(train)
    model = LinearRegression()
    model.fit(M, Y)
    
    x = train.ACTUAL.values[-D:]
    y = []
    for _ in range(300):
        p = model.predict(x.reshape(1, -1))
        y.append(p[0])
        x = np.hstack((x[1:], p))
    test["ACTUAL"] = y
    return test.copy()


# In[8]:


test_all = None
for i in range(1, 23):
    if test_all is None: 
        test_all = main(*select_ts(i))
    else:
        test_all = test_all.append(main(*select_ts(i)))


# In[9]:


test_all.set_index("ID", inplace=True)


# In[10]:


test_all.ACTUAL.to_csv("sub.csv")

