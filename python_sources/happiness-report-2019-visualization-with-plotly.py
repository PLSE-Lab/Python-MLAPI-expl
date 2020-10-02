#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly_express as px

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/2019-world-happiness-report-csv-file/2019.csv')
df.head()


# In[ ]:


df.tail()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


# we observe that there are no null values and also see the data types


# In[ ]:


df.describe()


# In[ ]:


#get the score of top 10 ranking countries and plot them using Plotly


# In[ ]:


top_10 =df.iloc[ 0:10, 0:3]
top_10


# In[ ]:


fig = px.pie(top_10, values='Score', names='Country or region', color_discrete_sequence=px.colors.sequential.RdBu, 
             title='Top 10 Country and their score',
             hover_data=['Overall rank'], labels={'Overall rank':'Overall rank'})
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# In[ ]:


df1 = df.iloc[0:20]
df1.head()


# In[ ]:


df1.shape


# In[ ]:


fig = px.pie(df1, values='GDP per capita', names='Country or region', 
             color_discrete_sequence=px.colors.sequential.RdBu, 
             title='Top 20 Country with GDP score',
             hover_data=['Overall rank'])
             #labels={'Perceptions of corruption':'Perceptions of corruption'}) 
                                                                       
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# We observe that Finland ranks number one but its GDP is far below 16 countries. Hence GDP is not the sole determinent of a nation's happiness

# In[ ]:


fig = px.pie(df1, values='Perceptions of corruption', names='Country or region', 
             color_discrete_sequence=px.colors.sequential.dense, 
             title='Top 20 Country with their Corruption score',
             hover_data=['Overall rank'])
             #labels={'Perceptions of corruption':'Perceptions of corruption'}) 
                                                                       
fig.update_traces(textposition='inside', textinfo='value+label')
fig.show()


# The top ranking countries have low corruption and high GDP; but we also notice that 
# Perception of corruption feature has a greater say in determing the happiness of a nation
# High score of Perception of corruption indicates people have trust in the system, they feel secure
# where corruption is low. 
# Example: Denmark having lowest corruption 

# In[ ]:


df2 =  df.iloc[136:156]
df2.head()


# In[ ]:


df2.shape


# In[ ]:


fig = px.pie(df2, values='Perceptions of corruption'
                          , names='GDP per capita', 
             #color='GDP per capita',
             color_discrete_sequence=px.colors.sequential.matter, 
             title='Bottom 20 Country with their GDP and  Corruption score',
             hover_data=['Overall rank'], hover_name='Country or region',
             labels={'Country or region':'Country or region'}) 
                                                                       
fig.update_traces(textposition='inside', textinfo='label')
fig.show()


# The above plot shows the GDP along with the corruption perception, and despite of having a moderate GDP a nation 
# might not rank well due to the prevelance od Corruption like in case of Egypt, Syria and India
# Egypt having a high GDP 9% but there is corruption and hence it ranks low 137 out of 156 nations

# In[ ]:


data_ss = df[df['Overall rank']>=140]
fig = px.bar(data_ss, x='GDP per capita', y='Social support',
             hover_data=['Healthy life expectancy', 'Freedom to make life choices', 'Overall rank'],
             color='Country or region',
             title='Bottom 17 countries with their details' ,
             height=400)
fig.show()


# In[ ]:


data_top = df[df['Overall rank']<=20]
data_top.shape


# In[ ]:


fig = px.bar(data_top, x='GDP per capita', y='Social support',
             hover_data=['Healthy life expectancy', 'Freedom to make life choices', 'Overall rank'],
             color='Country or region',
             title='Top 20 countries with their details' ,
             color_discrete_sequence=px.colors.sequential.thermal,                          
             height=600)
fig.show()


# we notice that Social support, freedom to make choices,life expectancy are crucial determinants 
# in ranking of the country. The above two bar plots can be studied further and the figures can be compared among 
# the top and the bottom countries.
