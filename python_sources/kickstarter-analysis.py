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


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv",
                encoding = 'latin-1')


# In[ ]:


df.columns = [col_names.replace(' ','') for col_names in df.columns]


# In[ ]:


df.drop(['ID','Unnamed:13','Unnamed:14','Unnamed:15','Unnamed:16'],axis=1,errors = 'ignore',inplace = True)


# In[ ]:


df.sort_values(by = 'backers',ascending = False)


# In[ ]:


df[['main_category','pledged']].groupby('main_category').agg(['mean','count']).sort_values(by = ('pledged',  'mean'))


# In[ ]:


df.sort_values(by = 'pledged',ascending = False)[:5]


# In[ ]:


df.sort_values(by = 'backers',ascending = False)[:5]


# In[ ]:


df['pledged'].sum()/df['backers'].sum()


# In[ ]:


df['ano'] = [datez.split('-')[0] for datez in df['deadline']]


# In[ ]:


ano_df = df.groupby('ano').mean()
sns.lineplot(x = ano_df.index, y = ano_df['pledged'])
plt.grid()


# In[ ]:


#ISto esta errado, preciso excluir os 3 de baixo
df['state'].value_counts(normalize = True)


# In[ ]:


imp_states = df[(df['state'] == 'failed' ) | (df['state'] == 'successful') | (df['state'] == 'canceled')]
imp_states = imp_states.groupby(['main_category','state']).count()[['name']]

(imp_states / imp_states.reset_index().groupby(by = 'main_category').sum()).pivot_table(index = 'main_category',
                                                                                        columns = 'state',
                                                                                       values = 'name')
#imp_states / imp_states.groupby(level=0).sum()


# In[ ]:


sns.countplot(df['currency'])

