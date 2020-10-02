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


df = pd.read_csv('../input/Dataset_spine.csv')
df.head()


# In[3]:


df.dtypes


# In[5]:


df.Class_att.value_counts().plot.bar()


# In[6]:


df.groupby('Class_att').mean()


# In[7]:


from pandas.plotting import parallel_coordinates

 
parallel_coordinates(df, 'Class_att')


# In[9]:


sns.lmplot(data = df, x ='degree_spondylolisthesis' , y ='pelvic_tilt',hue = 'Class_att', fit_reg= False, markers = ['x', 'o'])


# In[13]:


sns.boxplot(  y ='pelvic_tilt',hue = 'Class_att', data=df)


# In[15]:


df.plot.box( by = 'Class_att')  


# In[16]:


df.plot.box( by = 'Class_att')  


# In[17]:


df.boxplot(by ='Class_att')


# In[ ]:




