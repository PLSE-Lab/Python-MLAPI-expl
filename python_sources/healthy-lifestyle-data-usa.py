#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


df = pd.read_csv('../input/2011.csv')


# In[4]:


df.head()


# In[5]:


df.dtypes


# In[7]:


df.GENHLTH.value_counts().sort_index().plot.bar() # Health Status


# In[8]:


df.PHYSHLTH.hist() # Sick days


# In[9]:


df.MENTHLTH.hist() # Sick days


# In[10]:


df.SEATBELT.value_counts().sort_index().plot.bar() # Health Status


# In[12]:


#df.FLUSHOT.value_counts().sort_index().plot.bar() # Health Status


# In[ ]:




