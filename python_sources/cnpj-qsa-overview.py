#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# Simple overview of the data set.

# ### Data Set Structure

# In[44]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
get_ipython().run_line_magic('pylab', 'inline')


# In[45]:


# Read data file
data_file = '../input/ReceitaFederal_QuadroSocietario.csv'
df = pd.read_csv(data_file, sep = '\t', encoding='utf-8')


# In[46]:


# Data set sample 
df.head() 


# ### Data Set Size

# In[47]:


# Data set size
len(df)


# ### NULL values by atribute

# In[48]:


# Number of null values in data set
df.isnull().sum()


# ### Total of companies headquarters by state

# In[49]:


col_df = {'nr_cnpj': 'qt_cnpj'}
agg_df = {'nr_cnpj': 'nunique'}
plot_data = df.groupby(['sg_uf'], as_index=False).agg(agg_df).rename(columns=col_df)
f, ax = plt.subplots(figsize=(10, 15))
ax = sns.barplot(data=plot_data
                ,x='qt_cnpj'
                ,y='sg_uf'
                ,color='b')
ax.set(xlabel='Qty. of CNPJs', ylabel='UF')
plt.show()


# In[50]:


plot_data


# ### Total of business partner by type

# In[51]:


col_df = {'nm_socio': 'qt_socios'}
agg_df = {'nm_socio': 'count'}
plot_data = df.groupby(['in_cpf_cnpj','ds_cpf_cnpj'], as_index=False)    .agg(agg_df)    .rename(columns=col_df)    .sort_values(by='qt_socios', ascending=False)
f, ax = plt.subplots(figsize=(10, 15))
ax = sns.barplot(data=plot_data
                ,x='qt_socios'
                ,y='ds_cpf_cnpj'
                ,color='b')
ax.set(xlabel='Business Partner Type'
      ,ylabel='Qty. of Business Partner')
plt.show()


# In[52]:


plot_data


# ### Total of business parter by role

# In[53]:


col_df = {'nm_socio': 'qt_socios'}
agg_df = {'nm_socio': 'count'}
plot_data = df.groupby(['cd_qualificacao_socio','ds_qualificacao_socio'], as_index=False)    .agg(agg_df)    .rename(columns=col_df)    .sort_values(by='qt_socios', ascending=False)
f, ax = plt.subplots(figsize=(10, 15))
ax = sns.barplot(data=plot_data
                ,x='qt_socios'
                ,y='ds_qualificacao_socio'
                ,color='b')
ax.set(xlabel='Qty. of Business Partner'
      ,ylabel='Business Partner Role')
plt.show()


# In[54]:


plot_data

