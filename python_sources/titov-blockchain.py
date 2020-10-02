#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


df = pd.read_csv("../input/train_data.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


print(df.size)


# In[ ]:


small_df = df[0:2609224]
small_df.tail()


# In[ ]:


writer = pd.ExcelWriter('output.xlsx')


# In[ ]:


small_df.to_excel(writer,'Sheet1')


# In[ ]:


writer.save()


# In[3]:


g = sns.FacetGrid(small_df, col='CENA_CLIENT')
g.map(plt.hist, 'REGION_AZS', bins=10)


# In[ ]:




