#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
filenames = check_output(["ls", "../input"]).decode("utf8").strip().split('\n')


# Any results you write to the current directory are saved as output.


# In[4]:


filenames


# In[5]:


dfs = []
for f in filenames:
    df = pd.read_csv('../input/' + f)
    dfs.append(df)


# In[6]:


dfs[0].head()


# In[7]:


dfs[1].head()


# In[8]:


dfs[2].head()


# In[9]:


dfs[3].head()


# In[12]:


dfs[0].columns
# It is annoying to have lots of white space in all the column names. Is there a quick way to clean overall? Some library?


# In[18]:


dfs[0].groupby(' State')['Measure Performance Rate'].mean().sort_values().plot.bar()


# In[ ]:




