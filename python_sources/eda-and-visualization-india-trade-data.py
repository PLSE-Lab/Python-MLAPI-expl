#!/usr/bin/env python
# coding: utf-8

# **Loading Libraries**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import squarify #TreeMap
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Data Loading**

# In[ ]:


etrade_df = pd.read_csv('../input/india-trade-data/2018-2010_export.csv')
itrade_df = pd.read_csv('../input/india-trade-data/2018-2010_import.csv')


# **Head and tail of data**

# In[ ]:


etrade_df.head()


# In[ ]:


etrade_df.tail()


# In[ ]:


itrade_df.head()


# In[ ]:


itrade_df.tail()


# In[ ]:


etrade_df.info()


# In[ ]:


itrade_df.info()


# In[ ]:


etrade_df.describe()


# In[ ]:


itrade_df.describe()


# **Cleanup**

# In[ ]:


etrade_df.isnull().sum()


# In[ ]:


itrade_df.isnull().sum()


# In[ ]:


etrade_df[etrade_df.value==0].head(5)


# In[ ]:


itrade_df[itrade_df.value==0].head()


# In[ ]:


etrade_df[etrade_df.country == "UNSPECIFIED"].head(5)


# In[ ]:


itrade_df[itrade_df.country == "UNSPECIFIED"].head(5)


# In[ ]:


# Replace the missing datas of the value column with their means grouped by the commodity

# export data
etrade_df["value"].fillna(etrade_df.groupby('Commodity')['value'].transform('mean'),inplace = True)
# import data
itrade_df["value"].fillna(itrade_df.groupby('Commodity')['value'].transform('mean'),inplace = True)


# In[ ]:


etrade_df['value'].isnull().sum()


# In[ ]:


itrade_df['value'].isnull().sum()


# In[ ]:


yearly_export = etrade_df.groupby('year')['value'].sum()
plt.figure(figsize=(15, 6))
ax = sns.lineplot(x=yearly_export.index,y=yearly_export)
ax.set_title('Total exporting in the last 8 years')


# In[ ]:


yearly_import = itrade_df.groupby('year')['value'].sum()
plt.figure(figsize=(15, 6))
ax = sns.lineplot(x=yearly_import.index,y=yearly_import)
ax.set_title('Total importing in the last 8 years')


# In[ ]:


importer_lists = []
for i in etrade_df['year'].unique():
    importer_lists.extend(etrade_df[etrade_df['year'] == i][['country','value']].groupby(['country']).sum().sort_values(by = 'value', ascending = False).iloc[0:3,:].index)


# In[ ]:


exporter_lists = []
for i in itrade_df['year'].unique():
    exporter_lists.extend(itrade_df[itrade_df['year'] == i][['country','value']].groupby(['country']).sum().sort_values(by = 'value', ascending = False).iloc[0:3,:].index)


# In[ ]:


from collections import Counter
favor_importer = Counter(importer_lists).most_common(3)

plt.figure(figsize=(12, 5))
for country, count in favor_importer:
    importer = etrade_df[etrade_df['country'] == country][['year','value','country']].groupby(['year']).sum()
    ax = sns.lineplot(x= importer.index, y= importer['value'])
ax.set_title('Top 3 favourite importers')


# In[ ]:


favor_exporter = Counter(exporter_lists).most_common(3)

plt.figure(figsize=(12, 5))
for country, count in favor_exporter:
    exporter = itrade_df[itrade_df['country'] == country][['year','value','country']].groupby(['year']).sum()
    ax = sns.lineplot(x= exporter.index, y= exporter['value'])
ax.set_title('Top 3 favourite exporters')


# In[ ]:


trade_partners = etrade_df[['country','value']].groupby(['country']).sum().sort_values(by = 'value', ascending = False).head()


# In[ ]:


plt.figure(figsize=(15, 6))
ax = sns.barplot(trade_partners.index, trade_partners.value, palette='Blues_d')
ax.set_title('Top 5 exporting partners')


# In[ ]:


trade_commodities = etrade_df[['Commodity','value']].groupby(['Commodity']).sum().sort_values(by = 'value', ascending = False).head(10)
trade_commodities


# In[ ]:


ax = sns.barplot(trade_commodities.value,trade_commodities.index, palette='Greens_d')
ax.set_title('Top 10 exporting commodities')


# In[ ]:


exporting_products = []
for i in etrade_df['year'].unique():
    exporting_products.extend(etrade_df[etrade_df['year'] == i][['Commodity','value']].groupby(['Commodity']).sum().sort_values(by = 'value', ascending = False).iloc[0:3,:].index)


# In[ ]:


from collections import Counter
favor_products = Counter(exporting_products).most_common(3)

plt.figure(figsize=(12, 5))
for product, count in favor_products:
    products = etrade_df[etrade_df['Commodity'] == product][['year','value','country']].groupby(['year']).sum()
    ax = sns.lineplot(x= products.index, y= products['value'])
    print(product)
ax.set_title('Top 3 favourite products')


# In[ ]:




