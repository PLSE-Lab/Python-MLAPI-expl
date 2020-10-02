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





# In[ ]:


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  

# we need to install mlxtend on anaconda prompt by typing 'pip install mlxtend'
from mlxtend.frequent_patterns import apriori  
from mlxtend.frequent_patterns import association_rules


# In[ ]:


data = pd.read_csv('../input/Pharmacy Data.csv') 
data.head()


# In[ ]:


basket = (data.groupby(['Ref','Article'])['Qty/Pk'].sum().unstack().reset_index()
          .fillna(0).set_index('Ref'))
basket.head()


# In[ ]:


def encoder(x):
    if x <= 0:
        return 0
    else:
        return 1
    
basket = basket.applymap(encoder)
basket.head()


# In[ ]:


basket.shape


# In[ ]:


items_together = apriori(basket, min_support = 0.001, use_colnames=True)
items_together.head()


# In[ ]:


rules = association_rules(items_together, metric='lift', min_threshold=1)
rules.head(10)


# In[ ]:




