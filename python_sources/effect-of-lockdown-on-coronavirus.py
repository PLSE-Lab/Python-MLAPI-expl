#!/usr/bin/env python
# coding: utf-8

# **Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.
# Most people who fall sick with COVID-19 will experience mild to moderate symptoms and recover without special treatment.**
# 
# **The virus that causes COVID-19 is mainly transmitted through droplets generated when an infected person coughs, sneezes, or exhales. These droplets are too heavy to hang in the air, and quickly fall on floors or surfaces.You can be infected by breathing in the virus if you are within close proximity of someone who has COVID-19, or by touching a contaminated surface and then your eyes, nose or mouth.**
# 
# *The objective of this notebook is to first visualize the spread of Coronavirus over time, and test out the hypothesis if one of the mitigation technique used by most countries(i.e Lockdown) had any effect on rate of transmission.*
# 
# This notebook is mainly divided into 3 phase:
# 1. [Visualize spread of Coronavirus over the time](#Spread-of-Coronavirus-over-the-time)
# 2. [Visualize Rate of Spread of virus in top 10 infected countries](#Spread-of-Coronavirus-over-time-for-top-10-infected-countries)
# 3. [Hypothesis testing if Lockdown had any effect on rate of transmission](#Did-Lockdown-effect-the-spread-of-coronavirus?)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px # plotly express
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
import pycountry
from geopy.geocoders import Nominatim
import os

# Input data files are available in the '/kaggle/input' or '../../../datasets/extracts/' directory.
file_input=['/kaggle/input','../../../datasets/extracts/']
files={}
for dirname, _, filenames in os.walk(file_input[0]):
    for filename in filenames:
        files[filename]=os.path.join(dirname, filename)
        print(filename)


# In[ ]:


# Get the Lockdown dates for all countries if exists
lockdown_df=pd.read_csv(files['countryLockdowndates.csv'])
lockdown_df['LockDown Date']=pd.to_datetime(lockdown_df['Date'],format='%d/%m/%Y')
lockdown_df.sort_values('LockDown Date',inplace=True)

# Reading time series data of confirmed cases all over the world
df=pd.read_csv(files['time_series_covid_19_confirmed.csv'])

# Pre-processing to remove negative data, if exists
df[df.columns[df.columns.str.contains('/20')]]=df[df.columns[df.columns.str.contains('/20')]].clip(lower=0)

# Column names as variables for ease-of-use
country_col='Country/Region'
confirmed_col='Confirmed Cases'

# Seeing a glimpse of how data looks and structured
df.head()


# In[ ]:


# Getting all countries 3-letter ISO codes for choropleth, using pycountry and geopy library [with country name and latitude/longitute provided in dataset]
locator = Nominatim(user_agent="myGeocoder")
def getIsoCodes(country_name,location):
    if pycountry.countries.get(name=country_name) is not None:
        return pycountry.countries.get(name=country_name).alpha_3
    elif pycountry.countries.get(alpha_2=country_name) is not None:
        return pycountry.countries.get(alpha_2=country_name).alpha_3
    else:
        location = locator.reverse(location)
        if 'address' in location.raw and'country_code' in location.raw['address'] and pycountry.countries.get(alpha_2=location.raw['address']['country_code'].upper()) is not None:
            return pycountry.countries.get(alpha_2=location.raw['address']['country_code'].upper()).alpha_3
        
        else:
            return ''

df['iso_codes']=df[[country_col,'Lat','Long']]        .apply(lambda record: getIsoCodes(record[country_col],', '.join(record[['Lat','Long']].astype(str).values)),axis=1)


# In[ ]:


# Transforming the dataframe by first grouping the Province/State to their respective Countires, and getting their sum 
# Transforming date columns to rows for further analysis, using pd.melt() for this purpose
confirmed_df= pd.melt(df[df.columns.difference(['Province/State','Lat','Long'])].groupby([country_col,'iso_codes']).sum().reset_index(),id_vars=[country_col,"iso_codes"], var_name="Date", value_name=confirmed_col)
confirmed_df= pd.merge(confirmed_df,lockdown_df[[country_col,'LockDown Date']].groupby(country_col).first(),left_on=country_col,right_on=country_col,how='left')

# Converting date to pandas datetime object for further sorting
confirmed_df['Date']=pd.to_datetime(confirmed_df['Date'])
confirmed_df.sort_values('Date',inplace=True)


# ## Spread of Coronavirus over the time

# In[ ]:


# Building animated choropleth to show the spread of coronavirus in the world
fig=px.choropleth(confirmed_df,
               locations='iso_codes',
               hover_name=country_col,
               animation_frame=confirmed_df['Date'].astype(str),
               color=confirmed_col,
               color_continuous_scale="Viridis",
               projection="natural earth",
               title="Confirmed Cases over the world"
              )
fig.show()


# In[ ]:


# Bar chart race for showing contries infection rate over time (showing 10 ten at a time)
plt.rcParams["animation.html"] = "jshtml"
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot()

colors = dict(zip(
    df[country_col].unique(),
    cm.rainbow(np.linspace(0,1,len(df[country_col].unique()))
)))

def draw_barchart(current_year):
    dff = confirmed_df[confirmed_df['Date'].eq(current_year)].sort_values(by=confirmed_col, ascending=True).tail(10)
    ax.clear()
    ax.barh(dff[country_col], dff[confirmed_col], color=[colors[x] for x in dff[country_col]])
    dx = dff[confirmed_col].max() / 200
    for i, (value, name) in enumerate(zip(dff[confirmed_col], dff[country_col])):
        ax.text(value-dx, i,     name,             size=14, weight=600, ha='right', va='bottom')
        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')
    ax.text(1, 0.4, current_year, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    ax.text(0, 1.06, confirmed_col, transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.15, 'Spread of Coronavirus over time',
            transform=ax.transAxes, size=24, weight=600, ha='left', va='top')
    ax.text(1, 0, 'by @Arnab Majumdar', transform=ax.transAxes, color='#777777', ha='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)
    plt.close()
    
# animating each frame of matplotlib chart using Funcanimation
animator = animation.FuncAnimation(fig, 
                                   draw_barchart, 
                                   frames=pd.date_range(start=confirmed_df['Date'].min(),
                                                        end=confirmed_df['Date'].max(),
                                                        freq='D').strftime('%Y-%m-%d'),
                                                        repeat=False,
                                                        cache_frame_data=False)
animator


# ## Spread of Coronavirus over time for top 10 infected countries

# In[ ]:


# top 10 countries
top_affected_countries= df.sort_values(confirmed_df['Date'].max().strftime('%-m/%-d/%y'),ascending=False)[country_col].iloc[:10].values

confirmed_df= confirmed_df[confirmed_df[country_col].isin(top_affected_countries)].sort_values('Date')

# Building a rangeslider log line chart to show the increase of confirmed cases in top 10 most infected countries
fig= px.line(confirmed_df, 
            color=country_col, 
            x='Date',
            y=confirmed_col,
            title='Confirmed Cases over time for top 10 infected countries')
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    yaxis=dict(type='log')
)
fig.show()


# ## Rate of spread of Coronavirus over time for top 10 infected countries

# In[ ]:


# Convertind value change in each day to Delta change (percentage change) using pandas in-built pct_change().
confirmed_pct_df=pd.concat([confirmed_df,confirmed_df.groupby([country_col])[confirmed_col].pct_change().rename('Percentage Change')*100],axis=1)

# Building a rangeslider log line chart to show delta change in confirmed cases in top 10 most infected countries
fig=px.line(confirmed_pct_df, 
            color=country_col, 
            x='Date',
            y='Percentage Change',
            title='Percentage Change each day for top 10 infected countries')
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    yaxis=dict(type='log',ticksuffix='%')
)
fig.show()


# ## Did Lockdown effect the spread of coronavirus?
# 
# *To answer that, we need to evaluate the rate of spread before and after lockdown. Average delta change and deviation from delta change mean will tell us the effect of lockdown.*
# 
# > Lower mean and low standard deviation will tell if the lockdown was successful in reducing the spread.

# In[ ]:


# Replacing infinite values with finite values [pct_change from 0 to x will give inf value, so converting it to x*100%, to get some information out of it]
confirmed_pct_df['Percentage Change']=confirmed_pct_df[[confirmed_col,'Percentage Change']]    .apply(lambda x: x['Percentage Change'] if x['Percentage Change']!= np.inf else x[confirmed_col]*100,axis=1)

# after adding lockdon daate to dataframe,usimg it to find for each record, if the record is before or after lockdown
confirmed_pct_df['After LockDown']=(confirmed_pct_df['Date']>confirmed_pct_df['LockDown Date']).astype(str)

# grouping by country,lockdown status to find mean and standard deviation before and after lockdown for top 10 countries
Mean_Median_Confirmed_df=confirmed_pct_df[[country_col,'After LockDown','Percentage Change']]    .groupby([country_col,'After LockDown']).agg(['mean','std'])
Mean_Median_Confirmed_df.columns=Mean_Median_Confirmed_df.columns.droplevel(0)
Mean_Median_Confirmed_df.rename({'mean':'Mean','std':'Standard Deviation'},axis=1,inplace=True)
Mean_Median_Confirmed_df


# In[ ]:


Mean_Median_Confirmed_df=Mean_Median_Confirmed_df.reset_index()

# Creating grouped bar chart to analyse the difference between mean and std of pct_change before and aftre lockdown
fig=px.bar(Mean_Median_Confirmed_df,
       x=country_col,
       y='Standard Deviation',
       color='After LockDown',
       barmode='group',
       title='Standard Deviation Comparison of Percentage Change Before & After Lockdown for top 10 infected countries')
fig.show()
fig=px.bar(Mean_Median_Confirmed_df,
       x=country_col,
       y='Mean',
       color='After LockDown',
       barmode='group',
       title='Mean Comparison of Percentage Change Before & After Lockdown for top 10 infected countries')     
fig.show()


# ## Conclusion
# > The above graph shows Mean and Standard Deviation of delta change before and After Lockdown.
# 
# ### **This proves the hypothesis that the Lockdown has helped control the spread of Coronavirus, and reduced the chances of sudden spike in covid cases for top affected countries.**
