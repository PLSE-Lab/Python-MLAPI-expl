#!/usr/bin/env python
# coding: utf-8

# # Covid19 Visualizer
# 
# Visualizer of Covid19 [Confirmed Cases /Deaths / Recovered]

#     Kindly get the latest data from the below link: 
# https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 100)


# In[ ]:


sns.set_style('darkgrid')


# ## Import Data

# In[ ]:


path = "../input/novel-corona-virus-2019-dataset/covid_19_data.csv"
df_original = pd.read_csv(path)


# In[ ]:


df = df_original.copy()


# In[ ]:


df.head(5)


# In[ ]:


df.describe()


# In[ ]:


df['Country/Region'].unique()


# In[ ]:


df.dtypes


# In[ ]:





# ## Sort Values / Filter

# In[ ]:


df = df.sort_values(by=['Confirmed'],ascending=False)
df.head()


# In[ ]:


italy_df = df.loc[df['Country/Region'] == 'Italy']
italy_df.tail()


# ## Filter by Country

# In[ ]:


df.head()


# In[ ]:


df_percountry = df.groupby(
    ['Country/Region','ObservationDate']).agg(
    {'Confirmed': 'sum','Deaths': 'sum', 'Recovered': 'sum'})
df_percountry.head()


# In[ ]:


df_percountry = df_percountry.reset_index().sort_values(by=['ObservationDate'],ascending=False)


# In[ ]:


df_percountry.head(5)


# In[ ]:


country = 'Saudi Arabia'


# In[ ]:


df_singlecountry =  df_percountry.loc[df_percountry['Country/Region'] == country]


# In[ ]:


df_singlecountry.head()


# ## Visualize for certain country

# In[ ]:


data = df_singlecountry
x = 'ObservationDate'
y = 'Confirmed'
d = 'Deaths'
r = 'Recovered'


# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.lineplot(x=data[x], y=data[y], err_style='band')
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases in X')

plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.lineplot(x=data[x], y=np.log1p(data[y]), err_style='band')
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases in X [Log+1 Transformed]')

plt.show()


# ### _Melting_

# In[ ]:


ids= ['ObservationDate', 'Country/Region']
values= ['Confirmed','Deaths','Recovered']


# In[ ]:


df_melted = pd.melt(df_singlecountry, id_vars=ids, value_vars=values)


# In[ ]:


df_melted.head()


# In[ ]:


data = df_melted
x = 'ObservationDate'
y = 'value'
z = 'variable'


# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.lineplot(data=data, x=x, y=y, hue=z, err_style='band')
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases in X')

plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.lineplot(data=data, x=x, y=np.log1p(data[y]), hue=z, err_style='band')
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases in X')

plt.show()


# ## Visualize UAE data

# In[ ]:


uae_df = df.loc[df['Country/Region'] == 'United Arab Emirates']


# In[ ]:


uae_df.head()


# In[ ]:


uae_df.tail()


# In[ ]:


sns.set_style('darkgrid')


# In[ ]:


data = uae_df
x = 'ObservationDate'
y = 'Confirmed'


# In[ ]:


data = uae_df
x = 'ObservationDate'
y = 'Deaths'


# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.lineplot(x=data[x], y=data[y], err_style='band')
ax = sns.lineplot(x=data[x], y=data['Recovered'])
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases in the UAE')

plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.lineplot(data=data, x=x, y=y, err_style='band')
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases in the UAE')

plt.show()


# In[ ]:


ids= ['ObservationDate', 'Country/Region']
values= ['Confirmed','Deaths','Recovered']


# In[ ]:


df_melted = pd.melt(uae_df, id_vars=ids, value_vars=values)


# In[ ]:


df_melted.head()


# In[ ]:


data = df_melted
x = 'ObservationDate'
y = 'value'
z = 'variable'


# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.lineplot(data=data, x=x, y=y, hue=z)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases in the UAE')

plt.show()


# ### Log Transformed Values

# In[ ]:


df_melted['logvalue'] = np.log1p(df_melted['value'])


# In[ ]:


df_melted.head()


# In[ ]:


data = df_melted
x = 'ObservationDate'
y = 'logvalue'
z = 'variable'


# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.lineplot(data=data, x=x, y=y, hue=z)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases/deaths/recovered in the UAE [Log+1 Transformed]')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df_percountry.unstack('Country/Region')


# In[ ]:




