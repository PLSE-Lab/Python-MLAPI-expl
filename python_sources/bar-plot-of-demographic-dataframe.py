#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


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


# In[3]:


df = pd.read_csv('../input/cv-valid-train.csv')


# In[4]:


df.head()


# In[12]:


df.dtypes


# In[9]:


df.age.unique()


# In[10]:


df.gender.unique()


# In[22]:


df.accent.value_counts().plot.bar()


# In[21]:


df.age.value_counts().plot.bar()


# In[20]:


df.gender.value_counts().plot.bar()


# In[11]:


df.accent.unique()


# In[13]:


df.duration.hist()


# In[16]:


df.duration.mean(skipna = True)


# In[17]:


df.duration.unique()


# In[19]:


df.duration.sample(10)


# In[ ]:


#Weird. Duration is all NaN. 

