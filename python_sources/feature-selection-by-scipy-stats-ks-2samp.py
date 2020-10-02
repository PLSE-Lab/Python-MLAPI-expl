#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/77537

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.drop(['target', 'ID'], inplace=True, axis=1)
test.drop(['ID'], inplace=True, axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


from tqdm import tqdm
from scipy.stats import ks_2samp
list_p_value =[]

for i in tqdm(train.columns):
    list_p_value.append(ks_2samp(test[i] , train[i])[1])

Se = pd.Series(list_p_value, index = train.columns).sort_values() 
list_discarded = list(Se[Se < .1].index)


# In[ ]:


len(list_discarded)


# In[ ]:


list_discarded


# In[ ]:




