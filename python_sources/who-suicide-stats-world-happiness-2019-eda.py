#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas_profiling
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# In[ ]:


dataset1 = pd.read_csv(f'/kaggle/input/who-suicide-statistics/who_suicide_statistics.csv')
dataset2 = pd.read_csv(f'/kaggle/input/world-happiness-report-2019/world-happiness-report-2019.csv')


# In[ ]:


dataset1.head(10)


# In[ ]:


dataset2.head(10)


# In[ ]:


dataset2.get_dtype_counts()


# In[ ]:


pandas_profiling.ProfileReport(dataset1)


# In[ ]:


pandas_profiling.ProfileReport(dataset2)


# In[ ]:


sns.set(style='whitegrid')
axe = sns.barplot(x='age',y='suicides_no',data=dataset1)
plt.xticks(rotation=90)


# In[ ]:


sns.set(style='whitegrid')
axe = sns.barplot(x='Country_(region)',y='Social_support',data=dataset2)
plt.xticks(rotation=90, size=3)


# In[ ]:


dataset1['e'] = pd.Series(np.random.randn(sLength), index=dataset1.index)

