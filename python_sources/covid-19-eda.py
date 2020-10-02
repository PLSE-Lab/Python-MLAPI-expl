#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing needed libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import datetime
import gc
from bs4 import BeautifulSoup as bs
import requests

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from plotnine import *

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import folium


# # Extracting data

# Get the list of all the available files from https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports

# In[ ]:


res = requests.get('https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports')    
soup = bs(res.text, 'lxml')   
files = soup.find_all('a',class_="js-navigation-open")

filenames = [file.text for file in files]
filenames_csv = [csv for csv in filenames if csv.endswith('.csv')]

final_df = pd.DataFrame()


# In[ ]:


for i in range(0,len(filenames_csv)):
    name = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_daily_reports/"+ str(filenames_csv[i])
    df = pd.read_csv(name, parse_dates= [2])
    df['Date'] = pd.to_datetime(filenames_csv[i].strip('.csv'))
    final_df = final_df.append(df)
    


# In[ ]:


final_df_backup = final_df.copy()
final_df.head()


# # Data Manipulation and Cleaning

# In[ ]:


final_df.shape


# In[ ]:


final_df.describe()


# In[ ]:


final_df.info()


# # Treating NULL Values

# In[ ]:


print("Before Treatment of NULL values \n")
print(final_df.isna().sum())


# In[ ]:


print("After Treatment of NULL values \n")
final_df[['Latitude', 'Longitude','Province/State']] = final_df[['Latitude', 'Longitude','Province/State']].fillna('NA')
final_df[['Confirmed', 'Deaths','Recovered']] = final_df[['Confirmed', 'Deaths','Recovered']].fillna(0)

print(final_df.isna().sum())


# In[ ]:


#Still infected
final_df['Still Infected'] = final_df['Confirmed'] - final_df['Deaths'] - final_df['Recovered']


# In[ ]:


cols = ['Date', 'Country/Region','Province/State','Confirmed','Deaths','Recovered','Still Infected']
final_df = final_df[cols]


# In[ ]:


final_df.head()


# In[ ]:


#Getting the records corresponding to the latest date

max_date = final_df['Date'].max()
final_df_latest = final_df[final_df['Date']== max_date]
final_df_latest.shape


# # EDA and Data visualization

# <H3> Global Spread </H3>

# In[ ]:


df_date_level = final_df.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Still Infected'].sum().reset_index()
df_date_level.sort_values(by= 'Date', inplace = True)

df_date_level['Virality'] = df_date_level['Confirmed'].pct_change()*100
df_date_level['Death Ratio'] = (df_date_level['Deaths']/df_date_level['Confirmed'])*100
df_date_level['Recovered Ratio'] = (df_date_level['Recovered']/df_date_level['Confirmed'])*100
df_date_level['Still Infected Ratio'] = (df_date_level['Still Infected']/df_date_level['Confirmed'])*100

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_date_level['Date'], y=df_date_level['Virality'],
                    mode='lines+markers',
                    name='Virality'))
fig.add_trace(go.Scatter(x=df_date_level['Date'], y=df_date_level['Death Ratio'],
                    mode='lines+markers',
                    name='Death%'))
fig.add_trace(go.Scatter(x=df_date_level['Date'], y=df_date_level['Recovered Ratio'],
                    mode='lines+markers', 
                    name='Recovered%'))

fig.add_trace(go.Scatter(x=df_date_level['Date'], y=df_date_level['Still Infected Ratio'],
                    mode='lines+markers', 
                    name='Still Infected%'))

fig.update_layout(
    title="Trend over Time",
    xaxis_title="Date",
    yaxis_title="Metric",
    font=dict(
        size=18
    )
)

fig.show()


# <H3> Country Level 

# In[ ]:


df_country_level = final_df_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Still Infected'].sum().reset_index()

df_country_level['Death%'] = (df_country_level['Deaths']/df_country_level['Confirmed'])*100
df_country_level['Recovered%'] = (df_country_level['Recovered']/df_country_level['Confirmed'])*100
df_country_level['Still Infected%'] = (df_country_level['Still Infected']/df_country_level['Confirmed'])*100

df_country_level['text'] = df_country_level['Country/Region'] + '<br>' +     'Confirmed Cases ' + df_country_level['Confirmed'].astype('str') +  '<br>' +     'Death % ' + df_country_level['Death%'].astype('str')+ '<br>' + ' Recovered % '  + df_country_level['Recovered%'].astype('str') 

fig = go.Figure(data=go.Choropleth(
    locations = df_country_level['Country/Region'],
    locationmode='country names',
    z = df_country_level['Confirmed'],
    text = df_country_level['text'], 
    autocolorscale=True,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
))

fig.update_layout(
    title_text='Total Confirmed Cases',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    )
)
fig.show()


# # Country and Date level

# In[ ]:


df_country_date = final_df.groupby(['Date','Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Still Infected'].sum().reset_index()

df_country_date['Virality'] = df_country_date.groupby(['Country/Region'])['Confirmed'].pct_change()*100
df_country_date['Death%'] = (df_country_date['Deaths']/df_country_date['Confirmed'])*100
df_country_date['Recovered%'] = (df_country_date['Recovered']/df_country_date['Confirmed'])*100
df_country_date['Still Infected%'] = (df_country_date['Still Infected']/df_country_date['Confirmed'])*100

df_country_date['size'] = df_country_date['Confirmed'].pow(0.5)
df_country_date['size_1'] = df_country_date['Deaths'].pow(0.5)
df_country_date['Date'] = df_country_date['Date'].dt.strftime('%m/%d/%Y')
df_country_date.sort_values('Date', ascending = True).reset_index()

fig = px.scatter_geo(df_country_date, locations="Country/Region", locationmode='country names', 
                     color="Confirmed", size='size', hover_name="Country/Region", 
                     range_color= [0, max(df_country_date['Confirmed'])+2], 
                     projection="natural earth", animation_frame="Date", 
                     title='Spread over time')
fig.show()

fig = px.scatter_geo(df_country_date, locations="Country/Region", locationmode='country names', 
                     color="Deaths", size='size_1', hover_name="Country/Region", 
                     range_color= [0, max(df_country_date['Deaths'])+2], 
                     projection="natural earth", animation_frame="Date", 
                     title='Deaths over time')
fig.show()


# In[ ]:




