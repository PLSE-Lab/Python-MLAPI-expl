#!/usr/bin/env python
# coding: utf-8

# [Covonavirus(COVID-19)](https://en.wikipedia.org/wiki/Coronavirus_disease_2019) is all over us. Organiations and Governments around the world are taking a number of policy actions to mitigate the spread.
# 
# It started in Wuhan,China and now it has spread to 150+ countries. Even countries smaller in size have been seeing large number of cases. Nobody knows cure to this disease but we are becoming more informed and smarter about actions that should be taken to minimize the impact.
# 
# From quarantining, to using masks and sanitizers and taking other necessary steps can be great precaution. 
# 
# In all this, there has been constant support from world community over data collection and processing to gather more insights about the outbreak and what are the factors affecting. Here, in this kernel, I have attempted to find the relation between socioeconomic indicators of countries, to the impact of virus on those countries.

# # Installing libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Plotting libraries
import matplotlib.pyplot as plt
import plotly.express as px

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Reading Data

# In[ ]:


data = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv',parse_dates=['Date'])
#print(len(data))
#print(data.dtypes)
data.head()


# In[ ]:


data.rename(columns={'Id': 'id',
                     'Province/State':'state',
                     'Country/Region':'country',
                     'Lat':'lat',
                     'Long': 'long',
                     'Date': 'date', 
                     'ConfirmedCases': 'confirmed',
                     'Fatalities':'deaths',
                    }, inplace=True)
data.head()


# In[ ]:


data = data.sort_values(by=['country','date'],ascending=[True,True])
#data.head()


# In[ ]:


# Sorting by country and date
data = data.groupby(['country','date'])['country','date','confirmed','deaths'].sum().reset_index()
#print(len(data))
data.head()


# In[ ]:


# Worldwide
worldwide = data.groupby(['date'])[['date','confirmed','deaths']].sum().reset_index()
worldwide.tail()


# # Plotting worldwide cases

# In[ ]:


fig = px.line(worldwide,'date','confirmed',title='Worldwide confirmed cases')
fig.show()

fig = px.line(worldwide,'date','confirmed',title='Worldwide confirmed cases (Log scale)',log_y = True)
fig.show()


# In[ ]:


# China
grouped_china = data[data.country=='China']

fig = px.line(grouped_china,'date','confirmed',title='China confirmed cases')
fig.show()

fig = px.line(grouped_china,'date','confirmed',title='China confirmed cases (Log scale)',log_y = True)
fig.show()


# In[ ]:


# USA
grouped_usa = data[data.country=='US']

fig = px.line(grouped_usa,'date','confirmed',title='USA confirmed cases')
fig.show()

fig = px.line(grouped_usa,'date','confirmed',title='USA confirmed cases (Log scale)',log_y = True)
fig.show()


# In[ ]:


# India
grouped_india = data[data.country=='India']

fig = px.line(grouped_india,'date','confirmed',title='India confirmed cases')
fig.show()

fig = px.line(grouped_india,'date','confirmed',title='India confirmed cases (Log scale)',log_y = True)
fig.show()


# # Socio-Economic Indicators angle

# In[ ]:


# Cases per population
pop = pd.read_csv('../input/countryinfo/covid19countryinfo.csv')
pop.head()


# In[ ]:


# Treat values
pop['pop'] = pop['pop'].str.replace(',','').fillna(0).astype(int)


# In[ ]:


# Treat missing values
# Check missing values per column
pop.isnull().sum()


# In[ ]:


# Fill missing with mean
pop['smokers'] = pop['smokers'].fillna(pop['smokers'].mean())


# In[ ]:


# And since this data is on 100 scale so making it %
pop['smokers'] = pop['smokers']/100
pop['urbanpop'] = pop['urbanpop']/100


# In[ ]:


# Clean population data
pop_data = pop[['country','pop','density','medianage','urbanpop','hospibed','smokers']]
pop_data.describe()


# In[ ]:


fig = px.bar(pop.sort_values(by="pop", ascending=False)[:10],'country','pop',title = "Population country wise")
fig.update_layout(xaxis_title = 'Country',yaxis_title = 'Population')
fig.show()


# In[ ]:


# Join main data with population data
data = data.merge(pop_data,how = 'left', on =['country'])
data.head()


# # Fraction of population as confirmed COVID19 cases

# In[ ]:


data['confirmed_norm'] = data['confirmed']/data['pop']
data['deaths_norm'] = data['deaths']/data['pop']
data.describe()


# In[ ]:


# Locations having more than 1% cases?


# In[ ]:


data[data.confirmed_norm>0.01].tail()


# # Now plotting normalized data

# In[ ]:


# Italy
grouped_italy = data[data.country=='Italy']

fig = px.line(grouped_italy,'date','confirmed_norm',title='Italy confirmed cases')
fig.layout.yaxis.tickformat = ',.2%'
fig.update_layout(xaxis_title = 'Date',yaxis_title = 'Confirmed cases (%)')
fig.show()

fig = px.line(grouped_italy,'date','confirmed_norm',title='Italy confirmed cases (Log scale)',log_y = True)
fig.layout.yaxis.tickformat = ',.2%'
fig.update_layout(xaxis_title = 'Date',yaxis_title = 'Confirmed cases (%)')
fig.show()


# **This is quite huge (Italy has close to 0.1% population affected**

# In[ ]:


# India
grouped_india = data[data.country=='India']

fig = px.line(grouped_india,'date','confirmed_norm',title='India confirmed cases')
fig.layout.yaxis.tickformat = ',.5%'
fig.update_layout(xaxis_title = 'Date',yaxis_title = 'Confirmed cases (%)')
fig.show()

fig = px.line(grouped_india,'date','confirmed_norm',title='India confirmed cases (Log scale)',log_y = True)
fig.layout.yaxis.tickformat = ',.5%'
fig.update_layout(xaxis_title = 'Date',yaxis_title = 'Confirmed cases (%)')
fig.show()


# **India is not much affected yet, when seen on % basis**

# In[ ]:


# India
grouped_china = data[data.country=='China']

fig = px.line(grouped_china,'date','confirmed_norm',title='China confirmed cases')
fig.layout.yaxis.tickformat = ',.3%'
fig.update_layout(xaxis_title = 'Date',yaxis_title = 'Confirmed cases (%)')
fig.show()

fig = px.line(grouped_china,'date','confirmed_norm',title='China confirmed cases (Log scale)',log_y = True)
fig.layout.yaxis.tickformat = ',.3%'
fig.update_layout(xaxis_title = 'Date',yaxis_title = 'Confirmed cases (%)')
fig.show()


# China is high in absolute but less in percantage impact

# # Seeing latest numbers of confirmed cases

# In[ ]:


data['max_date'] = data.groupby(['country'])['date'].transform('max')
data['max_date_flag'] = np.where(data['date'] == data['max_date'],True,False)
latest_data = data[data['max_date_flag']]
latest_data.tail()


# In[ ]:


# Bar plot of most affected countries (% wise)
fig = px.bar(latest_data.sort_values(by="confirmed_norm", ascending=False)[:10],'country','confirmed_norm',title = "Confirmed per 100 (Country wise)")
fig.layout.yaxis.tickformat = ',.1%'
fig.update_layout(xaxis_title = 'Country',yaxis_title = 'Confirmed cases (%)')
fig.show()


# In[ ]:


# Remove countries with population less than million and confirmed cases less than 100
fig = px.bar(latest_data[(latest_data['pop'] >= 1e6) & (latest_data['confirmed'] >= 100)].sort_values(by="confirmed_norm", ascending=False)[:10],'country','confirmed_norm',title = "Confirmed per 100 (Country wise)")
fig.layout.yaxis.tickformat = ',.2%'
fig.update_layout(xaxis_title = 'Country',yaxis_title = 'Confirmed cases (%)')
fig.show()


# *** This was quite informative, Eurppean countries have highest impact**

# In[ ]:


clean_data = latest_data[(latest_data['pop'] >= 1e6) & (latest_data['confirmed'] >= 100)]
clean_data.describe()


# # Correlation between Median Age and Confirmed %

# In[ ]:


fig = px.scatter(clean_data,'medianage','confirmed_norm',title = "Confirmed % vs Median Age",
                hover_name="country")
fig.layout.yaxis.tickformat = ',.2%'
fig.update_layout(xaxis_title = 'Median Age',yaxis_title = 'Confirmed cases (%)')
fig.show()


# In[ ]:


# Correlation between smokers and confirmed norm
fig = px.scatter(clean_data,'smokers','confirmed_norm',title = "Confirmed % vs % Smokers",
                hover_name="country")
fig.layout.yaxis.tickformat = ',.2%'
fig.update_layout(xaxis_title = 'Smokers %',yaxis_title = 'Confirmed cases (%)')
fig.show()


# In[ ]:


# Correlation between urbanpop vs confirmed_norm
fig = px.scatter(clean_data,'urbanpop','confirmed_norm',title = "Confirmed % vs Urban population %",
                hover_name="country")
fig.layout.yaxis.tickformat = ',.2%'
fig.update_layout(xaxis_title = 'Urban population %',yaxis_title = 'Confirmed cases (%)')
fig.show()


# In[ ]:


# Correlation between Hospital bed vs confirmed_norm
fig = px.scatter(clean_data,'hospibed','confirmed_norm',title = "Confirmed % vs Hospital bed per 1k",
                                hover_name="country")
fig.layout.yaxis.tickformat = ',.2%'
fig.update_layout(xaxis_title = 'No. hospital beds per 1k',yaxis_title = 'Confirmed cases (%)')
fig.show()


# In[ ]:


# Correlation between Population density vs confirmed_norm
fig = px.scatter(clean_data,'density','confirmed_norm',title = "Confirmed % vs Population Density",
                                hover_name="country")
fig.layout.yaxis.tickformat = ',.2%'
fig.update_layout(xaxis_title = 'Population density (sq km)',yaxis_title = 'Confirmed cases (%)')
fig.show()


# # Map plot

# # Confirmed cases over time

# In[ ]:


plot_data = data.copy()
plot_data['date'] = pd.to_datetime(plot_data['date']).dt.strftime("%Y-%b-%d")
plot_data['factor_size'] = plot_data['confirmed'].pow(0.5)


# In[ ]:


fig = px.scatter_geo(plot_data, locations="country", locationmode='country names', 
                     color="confirmed", size='factor_size', hover_name="country", 
                     range_color= [1, 1000], 
                     projection="natural earth", animation_frame="date", 
                     title='Coronavirus (COVID 19): Spread Over Time', color_continuous_scale="portland")
#fig.update(layout_coloraxis_showscale=False)
fig.show()


# # Percentage of population as confirmed cases

# In[ ]:


# Percentage wise
plot_data['factor_size_pc'] = plot_data['confirmed_norm'].fillna(0).pow(0.2)
plot_data['confirmed_pc'] = plot_data['confirmed_norm'].fillna(0)*100


# In[ ]:


fig = px.scatter_geo(plot_data, locations="country", locationmode='country names', 
                     color="confirmed_pc", size='factor_size_pc', hover_name="country", 
                     range_color= [1e-10, 0.001], 
                     projection="natural earth", animation_frame="date", 
                     title='Coronavirus (COVID 19): Spread Over Time (% of population as confirmed cases)', color_continuous_scale="portland")
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# * As we are seeing, European countries are most hit when we compare cases to population
# * Older Age, smoking, and high level of interaction (urban) has effect on contracting COVID19

# In[ ]:




