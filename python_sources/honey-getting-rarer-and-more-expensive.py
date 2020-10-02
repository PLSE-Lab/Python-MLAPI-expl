#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[3]:


df = pd.read_csv('../input/honeyproduction.csv')
#dg = pd.read_csv('../input/honeyraw_2008to2012.csv')


# In[4]:


df.hist()


# In[5]:


df.notnull().sum()


# In[6]:


df.shape


# In[7]:


df.head()


# In[13]:


df.groupby('year').priceperlb.mean().plot.line()
#Honey getting more expensive 




# In[12]:


df.groupby('year').yieldpercol.mean().plot.line()


#Honey yield per colony decreases


# In[11]:


df.groupby('year').numcol.sum().plot.line()



# In[10]:


df.groupby('year').totalprod.sum().plot.line()
#Honey total production decreases in time


# In[9]:


df.groupby('year').prodvalue.sum().plot.line()

#Honey total production value increases in time


# In[ ]:




