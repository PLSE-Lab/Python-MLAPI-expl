#!/usr/bin/env python
# coding: utf-8

# # Motivation
# 
# Most recently, we have seen some countries fight against COVID-19. This notebook will focus on ONE graph only and to see who is winning and who is yet to see the worst.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
from matplotlib.animation import FuncAnimation

# hide warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Datasets used
# 
# I will be using 2 datasets, one is the updated data on COVID19 ([Link](http://www.kaggle.com/imdevskp/corona-virus-report)) and the second one is the containment and mitigation strategies implemented by each country ([Link](http://www.kaggle.com/paultimothymooney/covid19-containment-and-mitigation-measures)).

# In[ ]:


df = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])
containment = pd.read_csv('../input/covid19-containment-and-mitigation-measures/COVID%2019%20Containment%20measures%202020-03-30.csv')


# # Data Cleaning

# In[ ]:


containment = containment[containment['Keywords'] != 'first case'] # I am not taking records of first patient
containment['category'] = containment.Keywords.str.replace('14 days,', '')
containment2 = containment[['Country', 'Date Start', 'Date end intended', 'Description of measure implemented', 'Keywords', 'category']]

containment2['Date Start'] = pd.to_datetime(containment2['Date Start'])
containment2 = containment2[containment2['Country'].notna()]
containment2.dropna(how = 'all', inplace = True)
containment2 = containment2[~containment2.Country.str.contains("US:")]
containment2['Country'] = containment2['Country'].str.replace('United States', 'US')

## Here I am taking the first step taken by each country
contain_m = containment2.sort_values(by = ['Country','Date Start']).groupby('Country').head(1)

## Cleaning COVID19 data
df2 = df[df['Confirmed'] != 0]
df2['Province/State'] = df2['Province/State'].fillna(df2['Country/Region'])
df2['Active'] = df2['Confirmed'] - df2['Recovered']
df2 = df2.groupby(['Country/Region', 'Date']).sum().reset_index()

df2['newCases'] = df2.groupby(['Country/Region'])['Confirmed'].diff().fillna(df2['Confirmed'])
df2['newDeath'] = df2.groupby(['Country/Region'])['Deaths'].diff().fillna(df2['Deaths'])
df2['newRecovered'] = df2.groupby(['Country/Region'])['Recovered'].diff().fillna(df2['Recovered'])

## Merging with containment strategies data
df2 = df2.merge(contain_m, left_on = 'Country/Region', right_on = 'Country', how = 'left')


# # The main graph: How countries are doing with COVID19
# 
# ## Graph Explained
# I am plotting total confirmed cases against the average of news cases from last 5 days. I was influenced to make this graph by a youtube video ([Link](http://www.youtube.com/watch?v=54XLXg4fYs))
# 
# **X-Axis (Confirmed Cases):** Total confirmed cases in that country
# 
# **Y-Axis (Last 5 days average of new cases):** Average of new cases from last 5 days. I am taking average because it tells me a better picture as to how a country is doing instead of using new cases for one day. It will also make the curve more robust to sudden rise and fall in new reported cases
# 
# **Log-axis:** Both X and Y are in log form

# In[ ]:


mask = list(df2.groupby('Country/Region')['Confirmed'].max().sort_values(ascending = False).index[:20]) # Only plotting top 20 countries

temp = df2.copy()
temp['avg_n_case'] = temp.groupby('Country/Region')['newCases'].rolling(5).mean().reset_index(0,drop=True).fillna(temp['newCases'])
temp = temp[temp['Country/Region'].isin(mask)]
temp['Date'] = temp.Date.dt.strftime('%Y-%m-%d')

fig = px.line(temp, x = 'Confirmed', y = 'avg_n_case', log_x = True, log_y = True, color = 'Country/Region')
fig.update_layout(
    title="Trajectory to COVID19 of top 20 countries with most confirmed cases",
    yaxis_title="Average of new cases from last 5 days",
    xaxis_title="Total confirmed cases",
    )
fig.show()


# # What the graph above is telling us?
# 
# Plotting the graph on log axes made it clear that almost all countries are going on the same path. So if you draw a line of best fit, it will have a clear trajectory.
# 
# There are two countries namely China and South Korea who are way below the trend. It seems like these countries fough COVID19 and now they are winning the war agaist it.
# 
# **What about US:** US is currently going at a very fast pace followed by Spain anf Italy. It will take some time before the growth slows down. France and Iran are also not doing so good.

# # Making a dynamic plot to see trend with time

# In[ ]:


df3 = df.copy()
df3['Province/State'] = df3['Province/State'].fillna(df3['Country/Region'])
df3['Active'] = df3['Confirmed'] - df3['Recovered']
df3 = df3.groupby(['Country/Region', 'Date']).sum().reset_index()

df3['newCases'] = df3.groupby(['Country/Region'])['Confirmed'].diff().fillna(df3['Confirmed'])
df3['newDeath'] = df3.groupby(['Country/Region'])['Deaths'].diff().fillna(df3['Deaths'])
df3['newRecovered'] = df3.groupby(['Country/Region'])['Recovered'].diff().fillna(df3['Recovered'])

mask1 = list(df3.groupby('Country/Region')['Confirmed'].max().sort_values(ascending = False).index[:20])

temp2 = df3.copy()
temp2['avg_n_case'] = temp2.groupby('Country/Region')['newCases'].rolling(5).mean().reset_index(0,drop=True).fillna(temp['newCases'])
temp2 = temp2[temp2['Country/Region'].isin(mask1)]
temp2['Date'] = temp2.Date.dt.strftime('%Y-%m-%d')

fig = px.scatter(temp2, x = 'Confirmed', y = 'avg_n_case', log_x = True, log_y = True, text="Country/Region", size = 'Deaths',
             range_x=[0.1,800000], range_y=[0.1,80000], animation_frame = 'Date', color = 'Country/Region')
fig.update_traces(textposition='top center', showlegend = False)
fig.update_layout(
    title="Trajectory to COVID19 of top 20 countries with most confirmed cases (Size of marker = Total Deaths)",
    yaxis_title="Average of new cases from last 5 days",
    xaxis_title="Total confirmed cases",
    )
fig.show()


# # First step taken by winning and yet-to-win countries
# 
# **China** took the first step on 25h Dec 2019 by isolating the first case ("Medical staff in two Wuhan hospitals were suspected of contracting the virus; first reports of isolation being used")
# 
# **South** Korea took a radical step of implementing new testing strategy on 30th Jan 2020 ("Implemented new testing method (Real Time- RT PCR)- gene amplification method allows to test COVID-19 virus and receive results within 6 hours. Will distribute kits across private hospitals"
# 
# **United States** initially did not take it serously and only issued a warning against people coming out of Wuhan on 7th Jan 2020 ("United States CDC issues travel warning to for travellers to Wuhan for cluster of pneumonia cases of unknown etiology")

# In[ ]:


pd.set_option('display.max_colwidth', -1)
mask3 = ['US', 'China', 'South Korea', 'Italy', 'Spain', 'United Kingdom']
first_step = df2[df2['Country/Region'].isin(mask3)].        drop_duplicates(['Country/Region', 
                         'Keywords'], keep = 'last')[['Country/Region', 'Date', 
                                                      'Date Start', 'Description of measure implemented', 'category']]
first_step['Days since first step'] = abs(first_step['Date Start'] - first_step['Date'])
first_step


# # Top strategies used by China

# In[ ]:


china_best = containment2[containment2['Country'] == 'China']['Keywords'].value_counts().head(10)
china_best


# # Top strategies used by South Korea

# In[ ]:


korea_best = containment2[containment2['Country'] == 'South Korea']['Keywords'].value_counts().head(10)
korea_best


# # Final remarks
# 
# Even though, there is a lot of inclination towards "*self-distancing*", from China and South Korea, we can learn that **we have to be more aggressive in terms of identifying and immediately isolating people who are exhibiting symptoms of COVID-19.**
