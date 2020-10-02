#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Introduction
# 
# * This kernel is a sort of practical documentation of plotly for myself, I will update it permanently by adding new datasets and plots.

# ## PLOTS
# 
# 1. [Treemap](#1)
# 1. [Barplot](#2)
# 1. [Scatter Geo & Scatter MAP](#3)
# 1. [Choropleth Maps](#4)
# 1. [Choropleth: Timeframe Animation](#5)
# 
# 1. [Area & Scatter Plots](#6) 
# 
#     * [Area Plot](#7) 
#     * [Scatter Plot with Facets](#8)
#     
#     
# 1. [Horizontal Barplot](#9)
# 1. [Bar Chart Race](#10)
# 1. [Subplots](#11)
# 1. [Daily Death Tolls](#12) 
# 
#     * [Daily Death Tolls by Date](#13) 
#     * [Daily Death Tolls by Country](#14)

# <b id =1>
#     
# # 1. Treemap

# <font color=blue>
# * Loading and preparing data for visualization

# In[ ]:


data = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')


# In[ ]:


data.isnull().sum()


# In[ ]:


data.rename(columns={'Province/State':'State', 'Country/Region':'Country'}, inplace=True)


# ### Only final situation

# In[ ]:


grouped = data.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

last = grouped[grouped.Date == max(grouped.Date)]


# In[ ]:


temp = last.melt(id_vars='Date', value_vars=['Confirmed', 'Deaths', 'Recovered'])
temp


# In[ ]:


import plotly.express as px

fig  =  px.treemap(temp, path = ['variable'], 
                   values = 'value',                 
                   height=225)

fig.data[0].textinfo = 'label+value'

fig.show()


# <b id=2>
# # Barplot

# In[ ]:


fig = px.bar(temp, x='variable', y='value', color='variable', height=500, width=900)
fig.update_layout(title = {'text':'Total Cases by State', 'x':0.5, 'y':1}, 
                  legend = {'title':{'text':'Status','font':{'size':20, 'color':'orange'}}})
fig.show()


# <b id =3 >
#     
# # Scatter Geo & Scatter MAP

# In[ ]:


countries = data[data.Date == max(data.Date)].groupby(['Country', 'Lat', 'Long'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
countries.sample()


# <font color=blue>
# * check whether latitudes and longitudes are within the ranges: (-90, 90),  (-180, 180)

# In[ ]:


countries.describe()


# In[ ]:


fig = px.scatter_geo(countries, lat='Lat', lon='Long',
                       hover_name='Country',
                       hover_data=['Recovered'],
                       color = 'Confirmed',
                       size='Deaths',
                       projection='orthographic', # or 'natural earth'
                       height= 700,
                       color_continuous_scale='inferno_r')

fig.update_layout(title={'text':'COVID-19 Case Map','x':0.5,'y':1})    
fig.show()


# In[ ]:


fig = px.scatter_mapbox(countries, lat='Lat', lon='Long',
                       hover_name='Country',
                       hover_data=['Recovered'],
                       color = 'Deaths',
                       size='Confirmed',
                       width=1200,
                       zoom=.75,    #  SCALE OLARAK ZOOM
                       title={'text': "Covid-19 - MAPBOX", 'y':0.9, 'x':0.5},
                       mapbox_style='stamen-watercolor')

fig.show()


# <b id = 4>
# # Choropleth Maps
# 
# <font color=blue>
# * To create a choropleth map, there is a need for **ISO alpha-3** codes of countries.
# * So I uploaded another dataset having these codes, and match the countries in our main dataset

# In[ ]:


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')

codes = df.groupby(['COUNTRY', 'CODE']).sum().reset_index()

codes.head()


# In[ ]:


code_list = []
for i in countries.Country:
    if i in codes.COUNTRY.values:
        code_list.append(codes[codes.COUNTRY == i].CODE.values[0])
    else:
        code_list.append(np.nan)
        
        
countries['codes'] = code_list


# <font color=blue>
# * Check null code values of countries

# In[ ]:


countries[countries.codes.isna()]


# <font color= blue>
#     
# * Codes of some sigfinicant countries cannot be coded due to the differences in the country names between 2 dataframes.
# * Therefore, I will fill these codes by hand.

# In[ ]:


countries[countries['Country']=='US'] = countries[countries['Country']=='US'].fillna('USA')
countries[countries['Country']=='South Korea'] = countries[countries['Country']=='South Korea'].fillna('KOR')
countries[countries['Country']=='Czechia'] = countries[countries['Country']=='Czechia'].fillna('CZ')
countries[countries['Country']=='Bahamas'] = countries[countries['Country']=='Bahamas'].fillna('BHS')
countries[countries['Country']=='Congo (Brazzaville)'] = countries[countries['Country']=='Congo (Brazzaville)'].fillna('COG')
countries[countries['Country']=='Congo (Kinshasa)'] = countries[countries['Country']=='Congo (Kinshasa)'].fillna('COD')
countries[countries['Country']=='Gambia'] = countries[countries['Country']=='Gambia'].fillna('GMB')
countries[countries['Country']=='Holy See'] = countries[countries['Country']=='Holy See'].fillna('VAT')
countries[countries['Country']=='North Macedonia'] = countries[countries['Country']=='North Macedonia'].fillna('MKD')
countries[countries['Country']=='Taiwan*'] = countries[countries['Country']=='Taiwan*'].fillna('TWN')
countries[countries['Country']=='Western Sahara'] = countries[countries['Country']=='Western Sahara'].fillna('ESH')


# In[ ]:


codes = countries[~countries.codes.isna()]

codes = codes.groupby(['codes','Country'])['Confirmed','Deaths','Recovered'].sum().reset_index()
codes.head()


# ### Plot

# In[ ]:


fig = px.choropleth(codes, locations=codes['codes'],
                    color=np.log(codes.Deaths),
                   hover_name='Country',
                   hover_data=['Confirmed', 'Recovered'],
                   color_continuous_scale='inferno_r',
                   projection='natural earth',
                   height=800,
                   title={'text': "COVID-19 Choroplet Map", 'y':0.95, 'x':0.5})

fig.update_layout(font={'size':20, 'color':'purple'})
fig.update(layout_coloraxis_showscale=False)
fig.show()


# <b id= 5>
# # Choropleth: Timeframe Animation
#     
# <font color=blue>
# * This time, we make an animation of corona cases in a coropleth map by date. 
# 
# * Choropleth maps don't take a datetime object.
# 
# * But in order to sort dates, we need to convert 'Date' into a datetime object, then convert back into string values.

# * Append iso-alpha codes

# In[ ]:


timeframes = data.groupby(['Date', 'Country'])['Confirmed','Deaths','Recovered'].sum().reset_index()

code_list = []
for i in timeframes.Country:
    if i in codes.Country.values:
        code_list.append(codes[codes.Country == i].codes.values[0])
    else:
        code_list.append(np.nan)


# In[ ]:


timeframes['codes'] = code_list

timeframes = timeframes[~timeframes.codes.isna()]


# * Make datetime object and sort dates

# In[ ]:


timeframes['Date'] = pd.to_datetime(timeframes['Date']).sort_values()
timeframes = timeframes.sort_values(by='Date').reset_index(drop=True)


# In[ ]:


dates= []
for i in timeframes.Date.values:
    dates.append(str(i).replace('-', '/').split('T')[0])
    
timeframes['Dates'] = dates


# ### Plot

# In[ ]:


fig = px.choropleth(timeframes,
                   locations='codes',
                   color=np.log(timeframes.Deaths),
                   color_continuous_scale='inferno_r',
                   hover_name='Country',
                   hover_data=['Confirmed', 'Deaths', 'Recovered'],
                   animation_frame='Dates',
                   projection='natural earth')

fig.update_layout(title={'text':'Spread of COVID-19 by Date',
                         'font':{'size':18, 'color':'red'},
                         'x':0.5, 'y':1})
#                   transition = {'duration': 10000})

# fig.data[0].hovertext = 'Country' + timeframes.Country
fig.update(layout_coloraxis_showscale=False)
fig.show()


# <b id =6>
# # Area & Scatter Plots

# In[ ]:


area = data.groupby('Date')['Confirmed','Deaths','Recovered'].sum().reset_index()

area = area.melt(id_vars='Date', value_vars=['Confirmed','Deaths','Recovered'])

area = area.sort_values(by='Date').reset_index(drop=True)


# <font color=blue>
# * For the textual date format in x-axis, there is a need to convert dates to datetime object

# In[ ]:


area['Date'] = pd.to_datetime(area['Date']).sort_values()


# <b id=7>
# ## Area Plot

# In[ ]:


fig  = px.area(area, x='Date', y='value', color='variable', width=1200, height=700) # , facet_col='variable')

fig.update_layout(title={'text':'Cases over Time', 'x':0.5, 'y':.95, 'font':{'size':20}})
fig.data[0].mode = 'markers'
fig.data[1].mode = 'markers'
fig.data[2].mode = 'markers+lines'
fig.layout.legend ={'title':{'text':'Status','font':{'color':'blue', 'size':20}}, 'font':{'size':15}}
# fig.update_layout(legend = {'title':{'text':'CASES','font':{'color':'red', 'size':20}}, 'font':{'size':20}})

fig.show()


# <b id=8>
# ## Scatter Plot with Facets

# In[ ]:


fig  = px.scatter(area, x='Date', y='value', color='variable', width=1200, height=500,
                 facet_col='variable')

fig.update_layout(title={'text':'Cases over Time', 'x':0.5, 'y':1})
fig.data[0].update(mode='markers')
fig.data[1].update(mode='markers')
fig.data[2].update(mode='markers')

fig.show()


# <b id=9>
# # Horizontal Barplot

# <font color=blue>
# ### TOP 15 FINAL CASES

# In[ ]:


top_cases = data[data.Date == max(data.Date)].groupby('Country')['Confirmed', 'Deaths', 'Recovered'].sum().nlargest(columns=['Deaths'], n=15).sort_values(by='Deaths', ascending=False)
top_cases = top_cases.reset_index()
top_cases


# In[ ]:


fig = px.bar(top_cases, y='Country', x='Deaths', color='Country', orientation='h')

fig.show()


# <b id=10>
# 
# # Bar Chart Race 

# In[ ]:


temp = data.groupby(['Date','Country'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()


# In[ ]:


top_countries = data.groupby('Country')['Confirmed', 'Deaths', 'Recovered'].sum().nlargest(columns=['Deaths'], n=15).reset_index()['Country'].values


# In[ ]:


tf = pd.DataFrame(columns = temp.columns.values)

for c in top_countries:
    tf = tf.append(temp[temp.Country == c])


# In[ ]:


tf = tf.sort_values(by='Date').reset_index(drop=True)

tf['Date'] = pd.to_datetime(tf['Date']).reset_index(drop=True)

tf = tf.sort_values(by='Date').reset_index(drop=True)


# In[ ]:


dates = []

for i in tf.Date.values:
    dates.append(str(i).replace('-', '/').split('T')[0])
    
tf['Dates'] = dates
tf.head()


# In[ ]:


fig = px.bar(tf, y='Country', x='Deaths', animation_frame='Dates', color='Country', text=tf.Deaths, orientation='h', hover_name='Country', height=700)
fig.layout.title = {'text':'Death Tolls Caused by COVID-19 Over Time- aby','x':0.5, 'y':0.95}

fig.show()


# <font color= blue>
# 
# * I don't know if any other way to do a bar chart race in Plotly. But I couldn't sort the bars simultaneously with the animation as in [here](https://www.youtube.com/watch?v=2Uj1A9AguFs)
# 
# <font color= green>
# 
# I'll be so thankful if you help me about this

# <b id=11>
# 
# # Subplots

# In[ ]:


from plotly.subplots import make_subplots

fig = make_subplots(rows=1,cols=2, subplot_titles = ['Confirmed', 'Deaths'],
                                            specs = [[{"type": "choropleth"}, {"type": "choropleth"}]])

fig.add_trace(px.choropleth(codes, locations='codes', color=np.log(codes['Deaths']), hover_name='Country').data[0], 1,1)

fig.add_trace(px.choropleth(codes, locations='codes', color=np.log(codes['Confirmed']), hover_name='Country').data[0], 1,2)

fig.update(layout_coloraxis_showscale=False)

fig.show()


# In[ ]:


fig = make_subplots(1,2, subplot_titles=['CHOROPLETH', 'BAR'],
                   specs=[[{'type':'choropleth'}, {'type':'bar'}]])

fig.add_trace(px.choropleth(codes, locations='codes', color='Deaths').data[0], 1,1)
fig.add_traces(f2.data, rows=[1]*len(top_cases), cols=[2]*len(top_cases))
fig.update(layout_coloraxis_showscale=False)

fig.show()


# <b id=12>
# 
# # Daily Death Tolls

# In[ ]:


date_countries = data.groupby(['Date', 'Country'])['Confirmed', 'Deaths','Recovered'].sum().reset_index()

date_count = data.groupby('Date')['Confirmed', 'Deaths','Recovered'].sum().reset_index()


# In[ ]:


date_count['Date'] = pd.to_datetime(date_count['Date'])
date_count.sort_values(by='Date', inplace=True)
date_count.reset_index(drop=True, inplace=True)


# In[ ]:


date_countries['Date'] = pd.to_datetime(date_countries['Date'])
date_countries = date_countries.sort_values(by='Date')


# In[ ]:


temp = pd.DataFrame(columns=date_countries.columns)

for i in top_countries[:10]:
    temp = temp.append(date_countries[date_countries.Country == i])
    
temp.sort_values(by=['Date','Country'], inplace=True)
temp = temp.reset_index(drop=True)


# In[ ]:


daily_country = []
for idx, i in enumerate(temp.Date.unique()):
    for c in temp.Country.unique():
        if idx == 0:
            daily_country.append(temp[(temp.Date== i) & (temp.Country == c)].Deaths.values[0])
        else:
            daily_country.append(temp[(temp.Date== temp.Date.unique()[idx]) & (temp.Country == c)].Deaths.values[0] - temp[(temp.Date== temp.Date.unique()[idx-1]) & (temp.Country == c)].Deaths.values[0])


# In[ ]:


temp['Daily'] = daily_country
temp.tail(10)


# In[ ]:


daily_cases = []
for idx in date_count.index:
    if idx == 0:
        daily_cases.append(date_count.Confirmed[idx])
    else:
        daily_cases.append(date_count.Confirmed[idx] - date_count.Confirmed[idx-1])

date_count['Daily Cases'] = daily_cases
date_count.head()


# <b id=13>
# ## Daily Death Tolls by Date

# In[ ]:


fig = px.bar(date_count, x='Date', y='Daily Cases', width=1000)

fig.show()


# <b id=14>
# ## Daily Death Tolls by Country

# In[ ]:


fig = px.bar(temp, x='Date', y='Daily', width=500, height=1500, facet_row='Country', color='Country')

fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




