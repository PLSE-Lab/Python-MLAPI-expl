#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
trade_export = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')
trade_export.head()


# In[ ]:


trade_export.info()


# In[ ]:


trade_export.value.isna().value_counts()


# In[ ]:


trade_export['HSCode'] = trade_export.HSCode.astype('category')


# # Countries with most Exports

# In[ ]:


mst_export  = trade_export.groupby('country',as_index= False)['value'].agg({'Total':sum,'aveg' :np.mean})
mst_export.head()


# In[ ]:


mst_export = mst_export.sort_values(by = "Total", ascending=False)
mst_export.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,8))
plt.title('Countries with most Exports')
sns.barplot(x='country',y='Total',data= mst_export.head(), palette='Set3')


# # Most exported commodites

# In[ ]:


top_exp_com = trade_export.groupby('Commodity',as_index=False)['HSCode'].count().sort_values(by = 
                                                                               'HSCode', ascending = False)
top_exp_com.head()


# In[ ]:


plt.figure(figsize=(10,8))
plt.title('Most exported Commodities')
sns.barplot(y='Commodity',x='HSCode',data= top_exp_com.head(),palette='Set3')


# In[ ]:


trade_export[trade_export.value > 10000].count()


# # Most exported commodities by value

# In[ ]:


top_exp_com_val = trade_export.groupby('Commodity',as_index=False)['value'].count().sort_values(by = 
                                                                               'value', ascending = False)
top_exp_com_val.head()


# In[ ]:


plt.figure(figsize=(10,8))
plt.title('Most exported Commodities by value')
sns.barplot(y='Commodity',x='value',data= top_exp_com_val.head(),palette='Set3')


# # Imports Data

# In[ ]:


trade_import = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_import.csv")
trade_import.head()


# In[ ]:


trade_import.shape


# In[ ]:


trade_import.isna().any()


# In[ ]:


trade_import.duplicated().value_counts()


# In[ ]:


trade_import.drop_duplicates(inplace=True)
trade_import.shape


# In[ ]:


trade_export.info()


# # countries with most Imports  

# In[ ]:


mst_import  = trade_import.groupby('country', as_index= False)['value'].agg({'Total':sum,'aveg' :np.mean})
mst_import.head()


# In[ ]:


mst_import = mst_import.sort_values(by = "Total", ascending=False)
mst_import.head()


# In[ ]:


plt.figure(figsize=(10,8))
plt.title('Countries with most Imports')
sns.barplot(x='country',y='Total',data= mst_import.head(), palette='Set3')


# # most imported commodites

# In[ ]:


top_imp_com = trade_import.groupby('Commodity',as_index=False)['HSCode'].count().sort_values(by = 
                                                                               'HSCode', ascending = False)
top_imp_com.head()


# In[ ]:


plt.figure(figsize=(10,8))
plt.title('Most imported Commodities')
sns.barplot(y='Commodity',x='HSCode',data= top_imp_com.head(),palette='Set3')


# # Most exported commodities by value

# In[ ]:


top_imp_com_val = trade_import.groupby('Commodity',as_index=False)['value'].count().sort_values(by = 
                                                                               'value', ascending = False)
top_imp_com_val.head()


# In[ ]:


plt.figure(figsize=(10,8))
plt.title('Most Imported Commodities by value')
sns.barplot(y='Commodity',x='value',data= top_imp_com_val.head(),palette='Set3')


# # year wise

# In[ ]:


j=0
plt.figure(figsize=(25,20))

for i in np.arange(2010,2019):
    j+=1
    df = trade_export[trade_export.year == i].groupby(['country'],as_index=False)['value'].sum()
    df = df.sort_values(by = 'value',ascending =False).head()
    
    plt.subplot(3,3,j)
    #plt.ylim(0,50000)
    plt.title('Countries with most Exports for year %d' %i)
    sns.barplot(x = 'country', y = 'value', data =df)


# In[ ]:


df = trade_export.groupby(['country','year'],as_index=False)['value'].sum()
df.head()


# In[ ]:


df2 = df.pivot(index='country', columns='year', values= 'value')
df2.head()


# In[ ]:


df2['sum'] =  df2.sum(axis=1)
df2 = df2.sort_values(by = 'sum', ascending=False).head()
df2


# In[ ]:


df2 = df2.drop('sum', axis =1).T
df2


# In[ ]:


df2.plot(kind = 'line' ,figsize=(10,8), style = 'o-')
plt.title('Varation of Exports value with year')


# In[ ]:


df2.plot(kind = 'bar' ,figsize=(10,8), style = 'o-', title = 'Varation of Exports value with year')


# In[ ]:


j=0
plt.figure(figsize=(25,20))

for i in np.arange(2010,2019):
    j+=1
    df = trade_import[trade_import.year == i].groupby(['country'],as_index=False)['value'].sum()
    df = df.sort_values(by = 'value',ascending =False).head()
    
    plt.subplot(3,3,j)
    #plt.ylim(0,50000)
    plt.title('Countries with most Imports for year %d' %i)
    sns.barplot(x = 'country', y = 'value', data =df)


# In[ ]:


df3 = trade_import.groupby(['country','year'],as_index=False)['value'].sum()
df3.head()


# In[ ]:


df3 = df3.pivot(index='country', columns='year', values= 'value')
df3.head()


# In[ ]:


df3['Tot'] =  df3.sum(axis=1)
df3 = df3.sort_values(by = 'Tot', ascending=False).head()
df3


# In[ ]:


df3 = df3.drop('Tot',axis =1).T
df3


# In[ ]:


df3.plot(kind = 'line' ,figsize=(10,8), style = 'o-')
plt.title('Varation of Imports value with year')


# In[ ]:


df3.plot(kind = 'bar' ,figsize=(10,8))
plt.title('Varation of Imports value with year')


# In[ ]:




