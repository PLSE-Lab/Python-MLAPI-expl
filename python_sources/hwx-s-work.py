#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train_V2.csv")
test = pd.read_csv("../input/test_V2.csv")


# In[ ]:


train.columns


# In[ ]:


train.head()


# In[ ]:


feature = ["boosts","damageDealt","heals","kills","killStreaks","longestKill","walkDistance","weaponsAcquired"]
label = "winPlacePerc"


# In[ ]:


X_train = train[feature]


# In[ ]:


Y = train[label]


# In[ ]:


X_train.isnull().sum()


# In[ ]:


Y.isnull().sum()


# In[ ]:


Y = Y.fillna(0)


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,Y)


# In[ ]:


lm.score(X_train,Y)


# In[ ]:


X_test = test[feature]


# In[ ]:


lm.predict(X_test)


# In[ ]:


test[label] = lm.predict(X_test)


# In[ ]:


test[["Id","winPlacePerc"]].to_csv("Pubg data",index=False)


# In[ ]:




