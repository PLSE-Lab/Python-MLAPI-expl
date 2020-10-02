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
import matplotlib.pyplot as plt 

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input"]).decode("utf8").strip()


# In[2]:


df = pd.read_csv("../input/indian_liver_patient.csv") 
print(df.dtypes)
df.head()


# In[3]:


df.isnull().sum()


# We need to handle 4 missing values for Albumin_and_Globulin_Ratio. 

# In[10]:


df.Albumin_and_Globulin_Ratio.hist(bins = 20)


# In[4]:


df.describe()


# In[5]:


df = pd.get_dummies(df)


# In[6]:


df.head()


# Could not do some fancy pairGrid, because of discrete variables
# Now I can play with lots of pairGrid :) http://seaborn.pydata.org/generated/seaborn.PairGrid.html 

# In[8]:




g = sns.PairGrid(df, hue = 'Dataset')
g = g.map(plt.scatter)


# In[ ]:




