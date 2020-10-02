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


#Importing Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
import folium
from fbprophet import Prophet
from fbprophet.plot import plot_plotly


# In[ ]:


#Setting up plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot_mpl
import plotly.offline as py
init_notebook_mode(connected=True)


# In[ ]:


#importing data
ncv_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
ncv_deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_deaths.csv')
ncv_confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_confirmed.csv')
ncv_recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_recovered.csv')


# In[ ]:


ncv_data.head()


# In[ ]:


ncv_confirmed.head()


# In[ ]:


ncv_recovered.head()


# In[ ]:


ncv_confirmed_long = ncv_confirmed.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Confirmed' )
ncv_recovered_long = ncv_recovered.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Recovered')
ncv_deaths_long = ncv_deaths.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Deaths')
full_data = pd.concat([ncv_confirmed_long, ncv_deaths_long['Deaths'], ncv_recovered_long['Recovered']], axis=1, sort=False)


# In[ ]:


full_data.head()


# In[ ]:


#filling missing values with 0 if null values exists in columns of recovered, confirmed and deaths
full_data[['Confirmed', 'Deaths', 'Recovered']] = full_data[['Confirmed', 'Deaths', 'Recovered']].fillna(0)


# In[ ]:


full_data.head()


# In[ ]:


China = full_data[full_data['Country/Region'] == 'Mainland China']
Outsidechina = full_data[full_data['Country/Region'] != 'Mainland China']
full_latest = full_data[full_data['Date'] == max(full_data['Date'])].reset_index()
china_latest = full_latest[full_latest['Country/Region']=='Mainland China']
outsidechina_latest = full_latest[full_latest['Country/Region']!='Mainland China']

full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
outsidechina_latest_grouped = outsidechina_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()


# #EDA and Data visulisations

# In[ ]:


# confirmed cases country wise
conf_ncv = full_latest_grouped[['Country/Region', 'Confirmed']].sort_values(by = 'Confirmed', ascending=False).reset_index(drop=True)
conf_ncv.head()


# In[ ]:


#total no.of countries affected
print(len(conf_ncv))


# In[ ]:


# Affected Country/Region
print(f"Affected countries are : {full_data['Country/Region'].unique()}")
print(f"Total Affected countries are : {len(full_data['Country/Region'].unique())}")


# In[ ]:


fig = px.bar(full_data, x='Date', y='Confirmed', hover_data=['Province/State', 'Deaths', 'Recovered'], color='Country/Region')
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Confirmed bar plot for each Country',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig.update_layout(annotations=annotations)
fig.show()


# In[ ]:


#Provinces in china with most reported cases
china_rep = china_latest_grouped[['Province/State', 'Confirmed']].sort_values(by='Confirmed', ascending=False).reset_index(drop=True)
china_rep.head()


# In[ ]:


fig = px.bar(full_data.loc[full_data['Country/Region'] == 'Mainland China'], x='Date', y='Confirmed', hover_data=['Province/State', 'Deaths', 'Recovered'], color='Province/State')
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Confirmed bar plot for Mainland China',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig.update_layout(annotations=annotations)
fig.show()


# In[ ]:


#deaths as per country wise
cou_dea = full_latest_grouped[['Country/Region', 'Deaths']].sort_values(by='Deaths', ascending=False).reset_index(drop=True)
cou_dea = cou_dea[cou_dea['Deaths']>0]
print(cou_dea)


# In[ ]:


fig = px.bar(full_data, x='Date', y='Deaths', hover_data=['Province/State', 'Confirmed', 'Recovered'], color='Country/Region')
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Death bar plot for each country',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig.update_layout(annotations=annotations)
fig.show()


# **Maps visualisations**

# In[ ]:


fig = px.choropleth(full_latest_grouped, locations="Country/Region", locationmode='country names', 
                    color="Confirmed", hover_name="Country/Region", range_color=[1,50], color_continuous_scale="Sunsetdark", 
                    title='Countries with Confirmed Cases')
fig.update(layout_coloraxis_showscale=False)
fig.show()

fig = px.choropleth(full_latest_grouped[full_latest_grouped['Deaths']>0], locations="Country/Region", locationmode='country names',
                    color="Deaths", hover_name="Country/Region", range_color=[1,50], color_continuous_scale="Peach",
                    title='Countries with Deaths Reported')
fig.update(layout_coloraxis_showscale=False)
fig.show()

