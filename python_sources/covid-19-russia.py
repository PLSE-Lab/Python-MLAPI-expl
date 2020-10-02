#!/usr/bin/env python
# coding: utf-8

# # COVID 2019 - Russia

# Version: 14
# 
# 
# Last updated: 28.04.2020

# # Imports

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[ ]:


covid_19_data_complete = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")


# # Data preprocessing

# let's take a look on the data

# In[ ]:


covid_19_data_complete.head(5)


# select all data related to Russia

# In[ ]:


df = covid_19_data_complete[covid_19_data_complete['Country/Region'] == 'Russia']


# what we have:

# In[ ]:


df.head(5)


# ### Data selection

# In[ ]:


df_confirmed = df.drop(['Confirmed', 'Deaths', 'Recovered'], axis=1)
df_confirmed['count'] = df['Confirmed']
df_confirmed['type'] = 'Confirmed'


# In[ ]:


df_confirmed.head(5)


# In[ ]:


df_deaths = df.drop(['Confirmed', 'Deaths', 'Recovered'], axis=1)
df_deaths['count'] = df['Deaths']
df_deaths['type'] = 'Deaths'


# In[ ]:


df_deaths.head(5)


# In[ ]:


df_recovered = df.drop(['Confirmed', 'Deaths', 'Recovered'], axis=1)
df_recovered['count'] = df['Recovered']
df_recovered['type'] = 'Recovered'


# In[ ]:


df_recovered.head(5)


# Final df for EDA

# In[ ]:


frames = [df_confirmed, df_deaths, df_recovered]
df_for_eda = pd.concat(frames, sort=False)


# Convert the 'Date' column to the datetime format

# In[ ]:


df_for_eda['Date'] = pd.to_datetime(df_for_eda['Date'])


# # EDA

# Seaborn settings:

# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})


# Main plot

# In[ ]:


ax = sns.lineplot(x=df_for_eda['Date'], y=df_for_eda['count'], hue=df_for_eda['type'])
rotation = plt.setp(ax.get_xticklabels(), rotation=45)

