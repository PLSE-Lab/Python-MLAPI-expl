#!/usr/bin/env python
# coding: utf-8

# In[43]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# read in our data
shoes = pd.read_csv("../input/7210_1.csv", low_memory=False)
shoes.describe()
shoes.info()
# Any results you write to the current directory are saved as output.


# In[53]:


# Getting rid of missing data
shoes = shoes[["colors", "prices.amountMin", "prices.amountMax"]]
shoes.dropna(inplace=True)
shoes["midprices"] = (shoes["prices.amountMin"]+shoes["prices.amountMax"])/2
shoes.describe()


# In[69]:


test = shoes[["midprices","colors"]]
test.head()


# In[72]:


pink= shoes[shoes.colors =="Pink"]
notpink=shoes[shoes.colors!="Pink"]


# In[73]:


ttest_ind(pink.midprices,notpink.midprices,equal_var=False)


# In[77]:


pink["midprices"].describe()


# In[84]:


pink.midprices.plot.hist(bins=100, color="pink")
plt.xlim(0,500)
plt.legend(["Pink"])


# In[78]:


notpink["midprices"].describe()


# In[87]:


notpink.midprices.plot.hist(bins = 1000,color="orange")
plt.xlim(0,500)
plt.legend(["All colors except Pink"])


# In[ ]:




