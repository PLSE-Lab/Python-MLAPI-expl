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


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[5]:


docA = "The car is driven on the road"
docB = "The truck is driven on the highway"


# In[6]:


tfidf = TfidfVectorizer()


# In[7]:


response = tfidf.fit_transform([docA, docB])


# In[33]:


feature_names = tfidf.get_feature_names()
print(feature_names)


# In[32]:


print(response)


# In[29]:


print(response[1])


# In[21]:


for col in response.nonzero()[1]:
    print (feature_names[col], ' - ', response[0, col])


# In[ ]:




