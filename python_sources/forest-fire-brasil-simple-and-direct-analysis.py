#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv'
                     , encoding='latin1')


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.groupby('state')['number'].sum().sort_values(ascending=False)


# In[ ]:


plt.figure(figsize=(26,5))
sns.barplot(x=df['state'], y=df['number'], estimator=sum)


# In[ ]:


plt.figure(figsize=(26,5))
sns.barplot(x=df['year'], y=df['number'], estimator=sum)


# In[ ]:


heat = df.pivot_table(index='year', columns='state', values='number', aggfunc=sum)


# In[ ]:


sns.heatmap(heat)


# In[ ]:


meses={'Janeiro':1,'Fevereiro':2,'Marco':3,'Abril':4,'Maio':5,'Junho':6,'Julho':7,'Agosto':8,'Setembro':9,'Outubro':10,'Novembro':11,'Dezembro':12}


# In[ ]:


df['mes'] = df['month'].map(meses)


# In[ ]:


df.pivot_table(values='number',index='state', columns='mes', aggfunc=sum)


# In[ ]:


sns.heatmap(df.pivot_table(values='number',index='state', columns='mes', aggfunc=sum))


# In[ ]:




