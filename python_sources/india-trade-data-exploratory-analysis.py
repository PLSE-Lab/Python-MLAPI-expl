#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df_export = pd.read_csv("../input/india-trade-data/2018-2010_export.csv")
df_import = pd.read_csv("../input/india-trade-data/2018-2010_import.csv")


# ## Understanding the Dataset

# In[ ]:


df_export.head()


# In[ ]:


df_import.head()


# ## Missing Data?

# In[ ]:


df_export.isna().sum()


# In[ ]:


df_export['value'].fillna(0,inplace=True)


# In[ ]:


df_import.isna().sum()


# In[ ]:


df_import['value'].fillna(0,inplace=True)


# Lets check the other variables.

# In[ ]:


df_export[df_export['country']=='UNSPECIFIED'].head()


# In[ ]:


df_import[df_import['country']=='UNSPECIFIED'].head()


# Lets leave the UNSPECIFIED country for now.

# In[ ]:


df_export['year'].unique()


# In[ ]:


df_import['year'].unique()


# In[ ]:


df_export.duplicated().sum()


# In[ ]:


df_import.duplicated().sum()


# In[ ]:


df_import.drop_duplicates(keep="first",inplace=True)


# ## Exploring the Data

# Total exported and imported values. 

# In[ ]:


print('Total exportation value: {:.2f}'.format(df_export['value'].sum())) 
print('Total importation value: {:.2f}'.format(df_import['value'].sum())) 


# The 10 Countries with most Exports and Imports

# In[ ]:


pd.DataFrame(df_export['country'].value_counts()[:10])


# In[ ]:


pd.DataFrame(df_import['country'].value_counts()[:10])


# Most exported and imported Commodities

# In[ ]:


pd.DataFrame(df_export['Commodity'].value_counts()[:10])


# In[ ]:


pd.DataFrame(df_import['Commodity'].value_counts()[:10])


# Spendings in Export and Import

# In[ ]:


pd.DataFrame(df_export.groupby(df_export['Commodity'])['value'].sum().sort_values(ascending=False)[:10])


# In[ ]:


pd.DataFrame(df_import.groupby(df_import['Commodity'])['value'].sum().sort_values(ascending=False)[:10])


# Expensive Exports and Imports

# In[ ]:


df_export[df_export['value']==df_export['value'].max()]


# In[ ]:


df_import[df_import['value']==df_import['value'].max()]


# Value history throught the years

# In[ ]:


plt.figure(figsize=(10,10))
ax = sns.lineplot(data=df_export,x='year',y='value',err_style=None)
ax = sns.lineplot(data=df_import,x='year',y='value',err_style=None)
ax = plt.legend(['Export','Import'])

