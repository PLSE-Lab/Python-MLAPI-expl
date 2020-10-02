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
import seaborn as sns
import matplotlib.pyplot as plt


# Exports data

# In[ ]:


export = pd.read_csv('../input/india-trade-data/2018-2010_export.csv')
export.head()


# In[ ]:


export.info()


# In[ ]:


export['value'].fillna(export.groupby('Commodity')['value'].transform('mean'),inplace= True)


# In[ ]:


export['value'].isnull().sum()


# In[ ]:


plt.figure(figsize= (15,5))
sns.barplot('year','value', data= export)


# In[ ]:


exp_countries= export[['country','value']].groupby(['country']).sum().sort_values(by = 'value', ascending = False).head(20)


# In[ ]:


plt.figure(figsize= (20,5))
sns.barplot(exp_countries.index, exp_countries.value)


# In[ ]:


most_export= pd.DataFrame(export['Commodity'].value_counts().head(10))
most_export


# In[ ]:


by_country =export[['country','Commodity']].groupby(['country']).sum().sort_values(by = 'country', ascending = True).head(20)
by_country

