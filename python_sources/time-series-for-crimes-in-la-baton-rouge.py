#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


import fuzzywuzzy
from fuzzywuzzy import process

from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input/"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)


# In[2]:


df = pd.read_csv('../input/Baton_Rouge_Crime_Incidents.csv')


# In[4]:


df.dtypes


# In[3]:


df.head(5)


# In[8]:


df['date_parsed'] = pd.to_datetime(df['OFFENSE DATE'], infer_datetime_format=True)


# In[10]:


df['date_parsed'].sample(20)


# In[11]:


df['CRIME'].value_counts()


# In[12]:


df['COMMITTED'].value_counts()


# In[13]:


df['date_parsed'].value_counts().sort_values().plot.line()


# In[14]:


df['date_parsed'].value_counts().resample('M').sum().plot.line()


# In[16]:


df['date_parsed'].value_counts().resample('W').mean().plot.line()


# In[17]:


df.groupby('COMMITTED').count()


# In[ ]:




