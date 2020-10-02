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


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df  = pd.read_csv("../input/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe().transpose()


# In[ ]:


test_df.info()


# In[ ]:


test_df.describe().transpose()


# In[ ]:


table_df = pd.DataFrame({'train_df': [train_df.shape[0], train_df.shape[1]],
                      'test_df': [test_df.shape[0], test_df.shape[1]]}, index = ['rows','columns'])


# In[ ]:


table_df


# In[ ]:


df = pd.DataFrame({'A': [1,2,3],
                   'B': [4,5,6]}, index = ['a','b','c'])
df


# In[ ]:


table


# In[ ]:




