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


# In[34]:


df = pd.read_csv('../input/' + filenames)
df.drop(['ADDRESS','LOT', 'BLOCK', 'ZIP CODE', 'EASE-MENT','APARTMENT NUMBER', 'TAX CLASS AT PRESENT', 'BUILDING CLASS AT PRESENT'], axis = 1, inplace = True)
df['SALE DATE'] = pd.to_datetime(df['SALE DATE'], infer_datetime_format = True, errors = 'coerce')
df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'],   errors = 'coerce')
df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'],   errors = 'coerce')
df['GROSS SQUARE FEET'] = pd.to_numeric(df['GROSS SQUARE FEET'],   errors = 'coerce')
print(df.dtypes)
print(df.shape)
df.head()


# In[35]:


df.describe(exclude = 'O').transpose()


# In[36]:


df.describe(include = 'O').transpose()


# In[24]:


df.NEIGHBORHOOD.value_counts()


# In[23]:


df.BOROUGH.value_counts()


# In[3]:


#


# In[4]:





# In[8]:





# In[6]:


df.head()


# In[9]:


print(df.isnull().sum())


# In[16]:


df['GROSS SQUARE FEET'].median()


# In[19]:


df.describe(include ='O').transpose()


# In[17]:


df['LAND SQUARE FEET'].median()


# In[ ]:


dg = df.set_index('SALE DATE')


# In[ ]:


dg['SALE PRICE'].plot.line()


# In[ ]:


df.shape


# In[ ]:




