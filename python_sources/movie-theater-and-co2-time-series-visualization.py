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
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)


# In[2]:


df = pd.read_csv('../input/screening_times.csv')
dg = pd.read_csv('../input/TOF_CO2_data_30sec.csv')


# In[3]:


df.shape


# In[4]:


dg.shape


# In[5]:


df.head()


# In[6]:


dg.head()


# In[8]:


dg.dtypes.head()


# In[9]:


df.dtypes


# In[18]:


pd.to_datetime(dg.Time, format = '%m/%d/%Y %H:%M:%S')


# http://strftime.org 

# In[20]:


dg['Time_parsed'] = pd.to_datetime(dg.Time, infer_datetime_format=True)


# In[21]:


dg = dg.set_index('Time_parsed')


# In[22]:


dg.columns


# In[23]:


dg.CO2.plot()


# In[26]:


dg[dg.columns[2:5]].plot()


# In[27]:


df.head()


# In[28]:


df['begin_parsed'] = pd.to_datetime(df['begin'], infer_datetime_format = True)


# In[29]:


df.head()


# In[30]:


df = df.set_index('begin_parsed')


# In[31]:


df.head()


# In[32]:


df['number visitors'].plot.line()


# In[33]:


df['number visitors'].resample('D').sum().plot.line()


# In[34]:


df['number visitors'].resample('W').sum().plot.line()


# In[35]:


df['filled %'].plot.line()


# In[36]:


df.head()


# In[39]:


df.groupby('movie')['number visitors'].sum().sort_values(ascending = False)


# In[ ]:




