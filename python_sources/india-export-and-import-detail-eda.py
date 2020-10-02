#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xlrd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


exportdf = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_export.csv")
importdf = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_import.csv")


# In[ ]:


importdf.drop_duplicates(inplace=True)
exportdf.drop_duplicates(inplace=True)


# In[ ]:


print("Export Unqiue: "+str(len(exportdf['Commodity'].unique())))
print("Import Unqiue: "+str(len(importdf['Commodity'].unique())))


# In[ ]:


print("Export Value Sum: "+str(exportdf['value'].sum()))
print("Import Value Sum: "+str(importdf['value'].sum()))


# In[ ]:


print("Total Deficit of 10 year: "+str(exportdf['value'].sum()-importdf['value'].sum()))


# In[ ]:


growthImport = importdf.groupby('year').agg({'value':sum})
sns.barplot(y=growthImport.value,x=growthImport.index)


# In[ ]:


growthExport = exportdf.groupby('year').agg({'value':sum})
sns.barplot(y=growthExport.value,x=growthExport.index)


# In[ ]:


commodity = importdf[['value','Commodity']].groupby('Commodity').agg({'value':'sum'}).sort_values(by='value', ascending = False)[:10]
sns.barplot(y=commodity.index,x=commodity.value)


# In[ ]:


most_expensive = importdf[importdf.value>1000]
most_expensive1 = most_expensive.groupby(['country']).agg({'value':'sum'})
most_expensive1.sort_values(by='value',ascending=False)
most_expensive1


# In[ ]:


plt.figure(figsize=(15,5))
most_expensiveHSCode = most_expensive.groupby(['HSCode','country']).agg({'value':'sum'}).sort_values(by='value',ascending=False)[:15]
sns.barplot(most_expensiveHSCode.index,most_expensiveHSCode.value).set_xticklabels(sns.barplot(most_expensiveHSCode.index,most_expensiveHSCode.value).get_xticklabels(),rotation="90")


# In[ ]:




