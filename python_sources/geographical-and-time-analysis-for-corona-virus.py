#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis on Corona Virus
# 
# ## What is a Corona Virus? 
# 
# As listed on WHO website, Coronaviruses (CoV) are a large family of viruses that cause illness ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS-CoV) and Severe Acute Respiratory Syndrome (SARS-CoV). A novel coronavirus (nCoV) is a new strain that has not been previously identified in humans.  
# 
# Common signs of infection include respiratory symptoms, fever, cough, shortness of breath and breathing difficulties. In more severe cases, infection can cause pneumonia, severe acute respiratory syndrome, kidney failure and even death. 
# 
# ## Objective: 
# 
# Since we see that outbreak of Corona Virus is increasing Day by day, we can explore trends from the given data and try to predict future. 
# 
# ## Dataset Source: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset
# 
# 

# ## Exploratory Data Analysis
# 
# Let's perform EDA on the dataset.

# In[ ]:


# importing all necessary libraries
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pycountry
import plotly.graph_objects as go


# In[ ]:


# Reading the dataset
coronaVirus_df =  pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv",index_col='ObservationDate', parse_dates=['ObservationDate'])
coronaVirus_df.tail()


# In[ ]:


coronaVirus_df.shape


# ### Data Cleaning and Transformation
# 
# 1. Check for missing values and filling missing values
# 2. Change data type for Last Update column and modify other columns if required. 
# 3. Remove 'Sno' column as it is not required. 

# Checking missing values and transforming data

# In[ ]:


coronaVirus_df.isnull().values.any()


# In[ ]:


coronaVirus_df.isnull().sum()


# In[ ]:


#replacing null values in Province/State with Country names
coronaVirus_df['Province/State'].fillna(coronaVirus_df['Country/Region'], inplace=True)


# In[ ]:


coronaVirus_df.drop(['SNo'], axis=1, inplace=True)


# In[ ]:


coronaVirus_df.head()


# In[ ]:


#creating new columns for date, month and time which would be helpful for furthur computation
coronaVirus_df['year'] = pd.DatetimeIndex(coronaVirus_df['Last Update']).year
coronaVirus_df['month'] = pd.DatetimeIndex(coronaVirus_df['Last Update']).month
coronaVirus_df['date'] = pd.DatetimeIndex(coronaVirus_df['Last Update']).day
coronaVirus_df['time'] = pd.DatetimeIndex(coronaVirus_df['Last Update']).time


# In[ ]:


coronaVirus_df.head()


# In[ ]:


coronaVirus_df.rename(columns={"Country/Region": "Country", "Province/State": "State"} , inplace=True)


# > ### Latest Update on number of confirmed, reported and deaths across the globe****
# 
# We are trying to analyze number of cases reported.

# In[ ]:


# A look at the different cases - confirmed, death and recovered
print('Globally Confirmed Cases: ',coronaVirus_df['Confirmed'].sum())
print('Global Deaths: ',coronaVirus_df['Deaths'].sum())
print('Globally Recovered Cases: ',coronaVirus_df['Recovered'].sum())


# ![](http://)It is seen that total of 8786836 confirmed cases have been reported, 5591523 deaths have been confirmed and 23938502 people have sucessfully fought the virus and are showing signs of recovery. The data is from 22nd Jan to 4th March 2020. 
# 
# It is important to analyze latest scenario as per the last update so that we can predict numbers in future. 

# In[ ]:


coronaVirus_df[['Confirmed', 'Deaths', 'Recovered']].sum().plot(kind='bar', color = '#007bcc')


# ### Recovery % vs Death % across the globe
# 
# Let's check recovered% and death% across the globe

# In[ ]:


Recovered_percent = (coronaVirus_df['Recovered'].sum() / coronaVirus_df['Confirmed'].sum()) * 100
print("% of people recovered from virus: ",Recovered_percent)

Death_percent = (coronaVirus_df['Deaths'].sum()/coronaVirus_df['Confirmed'].sum()) * 100
print("% of people died due to virus:", Death_percent)


# In[ ]:


import plotly.graph_objects as go
grouped_multiple = coronaVirus_df.groupby(['ObservationDate']).agg({'Confirmed': ['sum']})
grouped_multiple.columns = ['Confirmed ALL']
grouped_multiple = grouped_multiple.reset_index()
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=grouped_multiple['ObservationDate'], 
                         y=grouped_multiple['Confirmed ALL'],
                         mode='lines+markers',
                         name='Confirmed',
                         line=dict(color='red', width=2)))
fig.show()


# ### Geographical Widespread of CoronaVirus
# 
# Using the given data, Here are few questions which we are going to answer
# 1. Total number of countries whch are affected by the virus
# 2. Number of confirmed, recovered, deaths cases reported Country wise
# 2. Number of confirmed cases reported State/Province wise
# 3. Top 5 Affected Countries
# 4. Top 5 countries which are unaffected.
# 5. Distribution of virus in India and US population. 

# In[ ]:


# Total Number Of countries which are affected by the virus

countries= coronaVirus_df['Country'].unique()
total_countries= len(countries)
print('Total countries affected:',total_countries)
print('Countries affected are:',countries)


# In[ ]:


# Number of confirmed cases reported Country wise

global_confirmed_cases = coronaVirus_df.groupby('Country').sum().Confirmed
global_confirmed_cases.sort_values(ascending=False)


# In[ ]:


global_death_cases = coronaVirus_df.groupby('Country').sum().Deaths
global_death_cases.sort_values(ascending=False)


# In[ ]:


global_recovered_cases = coronaVirus_df.groupby('Country').sum().Recovered
global_recovered_cases.sort_values(ascending=False)


# In[ ]:


#plotting graphs for total Confirmed, Death and Recovery cases
plt.rcParams["figure.figsize"] = (12,9)
ax1 = coronaVirus_df[['month','Confirmed']].groupby(['month']).sum().plot()
ax1.set_ylabel("Total Number of Confirmed Cases")
ax1.set_xlabel("month")

#ax2 = coronaVirus_df[['date','Deaths', 'Recovered']].groupby(['date']).sum().plot()
#ax2.set_ylabel("Recovered and Deaths Cases")
#ax2.set_xlabel("date")


# In[ ]:


# Let's look the various Provinces/States affected

data_countryprovince = coronaVirus_df.groupby(['Country','State']).sum()
data_countryprovince.sort_values(by='Confirmed',ascending=False)


# In[ ]:


# Top Affected countries

top_affected_countries = global_confirmed_cases.sort_values(ascending=False)
top_affected_countries.head(5)


# In[ ]:


# Finding countries which are relatively safe due to less number of reported cases
top_unaffected_countries = global_confirmed_cases.sort_values(ascending=True)
top_unaffected_countries.head(5)


# Above list are unaffected countries which means that relative to other countries, there are very less number of cases reported. These countries should take all measures to prevent spreading the virus.

# ### Plotting cases confirmed in China

# In[ ]:


#Mainland China
China_data = coronaVirus_df[coronaVirus_df['Country']=='Mainland China']
China_data


# In[ ]:


x = China_data.groupby('State')['Confirmed'].sum().sort_values().tail(15)


# In[ ]:


x.plot(kind='barh', color='#86bf91')
plt.xlabel("Confirmed case Count", labelpad=14)
plt.ylabel("State", labelpad=14)
plt.title("Confirmed cases count in China states", y=1.02);


# 1. > ### ****Geographical Distribution in India and US ****
# 
# > Now let's understand distribution of virus in US population

# In[ ]:


US_data = coronaVirus_df[coronaVirus_df['Country']=='US']
US_data


# In[ ]:


x = US_data.groupby('State')['Confirmed'].sum().sort_values(ascending=False).tail(20)
x


# In[ ]:


x.plot(kind='barh', color='#86bf91')
plt.xlabel("Confirmed case Count", labelpad=14)
plt.ylabel("States", labelpad=14)
plt.title("Confirmed cases count in US states", y=1.02);


# ### Coronavirus spread in India

# In[ ]:


India_data = coronaVirus_df[coronaVirus_df['Country']=='India']
India_data


# ## Time Series Analysis
# 
# It is important to understand correlation of time and cases reported. 

# In[ ]:


# Using plotly.express
import plotly.express as px

import pandas as pd

fig = px.line(coronaVirus_df, x='Last Update', y='Confirmed')
fig.show()


# In[ ]:



fig = px.line(coronaVirus_df, x='Last Update', y='Deaths')
fig.show()


# In[ ]:


import pandas as pd
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
                x=coronaVirus_df['date'],
                y=coronaVirus_df['Confirmed'],
                name="Confirmed",
                line_color='deepskyblue',
                opacity=0.8))

fig.add_trace(go.Scatter(
                x=coronaVirus_df['date'],
                y=coronaVirus_df['Recovered'],
                name="Recovered",
                line_color='dimgray',
                opacity=0.8))

fig.add_trace(go.Scatter(
                x=coronaVirus_df['date'],
                y=coronaVirus_df['Deaths'],
                name="Deaths",
                line_color='red',
                opacity=0.8))

# Use date string to set xaxis range
fig.update_layout(xaxis_range=['2020-01-22','2020-03-10'],
                  title_text="Cases over time")
fig.show()


# In[ ]:


import pandas as pd
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
                x=coronaVirus_df['date'],
                y=coronaVirus_df['Recovered'],
                name="Recovered",
                line_color='deepskyblue',
                opacity=0.8))

fig.add_trace(go.Scatter(
                x=coronaVirus_df['date'],
                y=coronaVirus_df['Deaths'],
                name="Deaths",
                line_color='red',
                opacity=0.8))

# Use date string to set xaxis range
fig.update_layout(xaxis_range=['2020-01-22 00:00:00','2020-03-10 23:59:59'],
                  title_text="Recovered vs Deaths over time in China")
fig.show()


# In[ ]:


import pandas as pd
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
                x=coronaVirus_df.time,
                y=coronaVirus_df['Confirmed'],
                name="Confirmed",
                line_color='deepskyblue',
                opacity=0.8))

# Use date string to set xaxis range
fig.update_layout(xaxis_range=['2020-01-31','2020-02-03'],
                  title_text="Confirmed Cases over time")
fig.show()


# As of now this is our EDA. We need to predict future cases and build models which will be coming soon. 

# ## References
# 
# https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset
# 
# https://www.who.int/health-topics/coronavirus
# 
# https://plot.ly/python/time-series/
# 
# https://plot.ly/python/bubble-maps/#base-map-configuration
# 
# https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea
# 
# https://en.wikipedia.org/wiki/Coronavirus
# 
# 
