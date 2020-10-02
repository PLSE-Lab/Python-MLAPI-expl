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


# In[5]:


df = pd.read_csv('../input/celebrity_deaths_4.csv', encoding = 'latin1')


# In[3]:


for size in range(1,9):
    length = 10 ** size
    with open('../input/celebrity_deaths_4.csv', 'rb') as rawdata:
        result = chardet.detect(rawdata.read(length))

    # check what the character encoding might be
    print(size, length, result)


# In[6]:


df.shape


# In[7]:


df.dtypes


# In[8]:


df.fame_score.hist()


# In[9]:


df.death_year.hist()


# In[10]:


df.age.hist()


# In[11]:


df.cause_of_death.value_counts().plot.bar()


# In[12]:


df.cause_of_death.value_counts().head(5)


# In[ ]:


# this column needs some text cleaning (fuzzy text)

