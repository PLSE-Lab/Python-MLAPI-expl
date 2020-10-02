#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[15]:


df = pd.read_json('../input/games_1512362753.8735218.json')


# In[3]:


dg = pd.read_json('../input/profiles_1512362725.022629.json')


# In[4]:


df.head()


# In[5]:


dg.head()


# In[6]:


df.dtypes


# In[10]:


df.player_team_score.hist()


# In[11]:


dg.current_salary.hist()


# In[12]:


dg.current_salary.unique()


# In[16]:


dg.current_salary.dtype


# in read_csv, we could have specified thousands = ',', but for read_json, I don't know.

# In[7]:


dg.dtypes


# In[9]:


dg.weight.hist()


# In[ ]:




