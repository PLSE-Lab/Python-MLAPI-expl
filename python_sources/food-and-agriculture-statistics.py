#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# In[ ]:


import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
print(os.listdir("../input"))


# In[ ]:


path = r'../input'                     # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f) for f in all_files)
df   = pd.concat(df_from_each_file, ignore_index=True)


# In[ ]:


df_India = df[df['country_or_area']=='India']


# In[ ]:


df_India


# In[ ]:


#pivoted = df_India.pivot_table(values='value',columns='year' )
df_India_year_value = df_India[['year','value']]


# In[ ]:


df_India_year_value


# In[ ]:


res = df_India_year_value.groupby('year')['value'].mean()
#a.round(decimals=2)
#a['value'].mean()

#df_India_1[df_India_1['year']==2007]['value'].mean()


# In[ ]:


res.round(decimals=2)


# In[ ]:


plt.plot(res.round(decimals=2),'r-*')


# In[ ]:




