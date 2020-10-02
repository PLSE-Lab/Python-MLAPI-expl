#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Coronaviruses are a large family of viruses that cause diseases ranging from the common cold to more serious diseases, such as Middle East Respiratory Syndrome (MERS) and Severe Acute Respiratory Syndrome (SARS).
# 
# A new coronavirus (COVID-19) was identified in 2019 in Wuhan, China. This is a new coronavirus that has not been previously identified in humans.

# # About this Analysis
# 
# It is an exploratory analysis of the Covid-19's progress in the world and Brazil.  It aims to discover relations, patterns, behaviors, trends and predictions, through answers to questions related to the data for be to analyzed.  The objective is also to observe the main characteristics of the data that reveal really objective and clear information, frequently by visual methods (graphs and tables), so that they are understood.
# 
# The Python programming language will be used to apply statistical techniques and machine learning algorithms through of procedure called Prophet.
# 
# Prophet is open source software released by Facebook's Data Science team. It a procedure for forecasting time series data. It works on non-linear models and fit to annual, weekly and daily seasonality. 
# 
# 
# 
#  * Assigned Questions:
# 
# 
# **In the world:**
# 
# * Q1 - What is the total number of confirmed, deaths and recoveries cases?
# 
# * Q2 - What is the growth rate of confirmed, deaths and recoveries cases?
# 
# * Q3 - What is the daily growth in the number of confirmed and deaths cases?
# 
# * Q4 - What are the numbers of confirmed, deaths, recovered, not recovered (active) in each country?
#  
# * Q5 - What is the lethality rate in each country?
#  
# * Q6 - Given the scenario presented, what is the forecast of the disease's progress in the world for confirmed and deaths cases?
#  
# * Q7 - What is the prediction to lethality rate?
# 
# 
# **In Brazil:**
# 
# * Q8 - What is the total confirmed, deaths and recoveries cases?
# 
# * Q9 - What is the growth rate of confirmed, deaths and recoveries?
# 
# * Q10 - What is the daily growth in the number of confirmed and deaths cases?
#  
# * Q11 - How many confirmed and deaths cases by region?
# 
# * Q12 - What is the lethality rate in each region?
#  
# * Q13 - How many confirmed and deaths cases by state?
# 
# * Q14 - What is the lethality rate in each state?
#  
# * Q15 - Given the scenario presented, what is the forecast of the disease's progress in Brazil for confirmed and deaths cases?
#  
# * Q16 - What is the prediction to lethality rate?
# 
# 
# **In Pernambuco - State of Brazil - Where i live**
# 
# * Q17 - How many confirmed and deaths cases?
# 
# * Q18 - What is the growth rate of confirmed and deaths?
# 
# * Q19 - What is the daily growth in the number of confirmed and deaths cases?
#  
# * Q20 - Given the scenario presented, what is the forecast of the disease's progress in Pernambuco for confirmed and deaths cases?
#  
# * Q21 - What is the prediction to lethality rate?

# # About these Datasets
# 
# These datasets explored in these analyzes below are provided by Johns Hopkins University, a renowned institution in the United States that is at the forefront of data collected worldwide about Covid-19.  I also collect data from the Kaggle platform, where it gathers users from all over the world collaborating with real data and from reliable sources.
# 
# All datasets explored have information with daily updates on the numbers of confirmed cases, deaths and recovery from Covid-19. Note that they are time series data and the numbers of cases on a given day are cumulative numbers.
# 
# * Sources (Datasets):
# 
# Datasets on github:
# 
# *     https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
# 
# *     https://github.com/niltontac/EspAnalise-EngDados/tree/master/data/Novel_Corona_Virus_2019_Dataset
# 
# *     https://github.com/niltontac/EspAnalise-EngDados/tree/master/data/covid19_brazil_data
# 
# Datasets on Kaggle:
# 
# *     https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset
# 
# *     https://www.kaggle.com/unanimad/corona-virus-brazil

# # Analyst: 
# 
# ### Nilton Thiago de Andrade Coura 
# 
# Brazil - Recife/Pernambuco
# 
# *     contact: niltontac@gmail.com
# 
# *     https://github.com/niltontac
# 
# *     https://www.linkedin.com/in/niltontac/

# # Covid-19 - Exploratory Analysis and Predictions
# 
# 
# 
# ![](https://i.ibb.co/txCZFvr/3-D-medical-animation-coronavirus-structure.jpg)

# **Import Libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import seaborn as sns
import plotly as py
import plotly.express as px

from fbprophet.plot import plot_plotly
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

import warnings
warnings.filterwarnings('ignore')


# **Loading datasets**
# 
# **Last Updates: July, 29, 2020**

# In[ ]:


covid19confirmed = pd.read_csv('../input/from-john-hopkins-university/time_series_covid19_confirmed_global.csv')

covid19recovered = pd.read_csv('../input/from-john-hopkins-university/time_series_covid19_recovered_global.csv')

covid19deaths = pd.read_csv('../input/from-john-hopkins-university/time_series_covid19_deaths_global.csv')

covid19 = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates=['ObservationDate', 'Last Update'])

covid19Brazil = pd.read_csv('../input/corona-virus-brazil/brazil_covid19.csv')


# In[ ]:


#Assigning last update:
last_update = '7/29/20'


# # Data processing and handling

# In[ ]:


#Checking the last 5 records to confirm when each dataset was updated:

print('covid19confirmed:')
print(covid19confirmed.tail())

###

print('covid19recovered:')
print(covid19recovered.tail())

###

print('covid19deaths:')
print(covid19deaths.tail())

###

print('covid19:')
print(covid19.tail())

###

print('covid19Brazil:')
print(covid19Brazil.tail())


# In[ ]:


#Rename column "ObservationDate" to 'Date'

covid19 = covid19.rename(columns={'ObservationDate' : 'Date'})


# In[ ]:


#Datasets (rows vs columns)

print('covid19confirmed:')
print(covid19confirmed.shape)

###

print('covid19recovered:')
print(covid19recovered.shape)

###

print('covid19deaths:')
print(covid19deaths.shape)

###

print('covid19:')
print(covid19.shape)

###

print('covid19Brazil:')
print(covid19Brazil.shape)

###


# In[ ]:


#Checking for null or missing data values in each dataset

print('covid19confirmed:')
print(pd.DataFrame(covid19confirmed.isnull().sum()))

###

print('covid19recovered:')
print(pd.DataFrame(covid19recovered.isnull().sum()))

###

print('covid19deaths:')
print(pd.DataFrame(covid19deaths.isnull().sum()))

###

print('covid19:')
print(pd.DataFrame(covid19.isnull().sum()))

###

print('covid19Brazil:')
print(pd.DataFrame(covid19Brazil.isnull().sum()))


# In[ ]:


#Some datasets have null or missings data values, then let's replace to "unknow" values

covid19confirmed = covid19confirmed.fillna('unknow') 
covid19recovered = covid19recovered.fillna('unknow')
covid19deaths = covid19deaths.fillna('unknow')
covid19 = covid19.fillna('unknow')


# # Plotly Visualizations: Exploratory Data Analysis and Predictions in the World and Brazil.

# # Worldwide:

# **Interactive Graph - Q01:**
# 
# **Total number of confirmed, death and recovered cases in the world**

# In[ ]:


all_cases_world = covid19.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum()
all_cases_world = all_cases_world.reset_index()
all_cases_world = all_cases_world.sort_values('Date', ascending=False)

fig = go.Figure()
fig.update_layout(title_text='Total number of confirmed, deaths and recovered cases in the World', 
                        xaxis_title='Period Date', 
                        yaxis_title='Total Cases', 
                        template='plotly_dark')

fig.add_trace(go.Scatter(x=all_cases_world['Date'],
                        y=all_cases_world['Confirmed'],
                        mode='lines+markers',
                        name='Global Confirmed',
                        line=dict(color='yellow', width=2)))

fig.add_trace(go.Scatter(x=all_cases_world['Date'],
                        y=all_cases_world['Deaths'],
                        mode='lines+markers',
                        name='Global Deaths',
                        line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=all_cases_world['Date'],
                        y=all_cases_world['Recovered'],
                        mode='lines+markers',
                        name='Global Recovered',
                        line=dict(color='green', width=2)))


fig.show()


# **Interactive Graph - Q02:**
# 
# **Global rate for growth confirmed, deaths and recovered cases**

# In[ ]:


global_rate = covid19.groupby(['Date']).agg({'Confirmed':['sum'],'Deaths':['sum'], 'Recovered': ['sum']})
global_rate.columns = ['Global_Confirmed', 'Global_Deaths', 'Global_Recovered']
global_rate = global_rate.reset_index()
global_rate['Increase_New_Cases_by_Day'] = global_rate['Global_Confirmed'].diff().shift(+1)
global_rate['Increase_Deaths_Cases_per_day']=global_rate['Global_Deaths'].diff().shift(+1)

#Calculating rates
#Lambda function
global_rate['Global_Deaths_rate_%'] = global_rate.apply(lambda row: ((row.Global_Deaths)/(row.Global_Confirmed))*100, axis=1).round(2)
global_rate['Global_Recovered_rate_%'] = global_rate.apply(lambda row: ((row.Global_Recovered)/(row.Global_Confirmed))*100, axis=1).round(2)
global_rate['Global_Growth_rate_%'] = global_rate.apply(lambda row: row.Increase_New_Cases_by_Day/row.Global_Confirmed*100, axis=1).round(2)
global_rate['Global_Growth_rate_%'] = global_rate['Global_Growth_rate_%'].shift(+1)
global_rate['Global_Growth_Deaths_rate_%']=global_rate.apply(lambda row: row.Increase_Deaths_Cases_per_day/row.Global_Confirmed*100, axis=1).round(2)
global_rate['Global_Growth_Deaths_rate_%']=global_rate['Global_Growth_Deaths_rate_%'].shift(+1)

fig = go.Figure()
fig.update_layout(title_text='Global rate of growth confirmed, deaths and recovered cases (%)',
                        xaxis_title='Period Date', 
                        yaxis_title='Rate', 
                        template='plotly_dark')

fig.add_trace(go.Scatter(x=global_rate['Date'],
                        y=global_rate['Global_Growth_rate_%'],
                        mode='lines+markers',
                        name='Global Growth Confirmed rate %',
                        line=dict(color='yellow', width=2)))

fig.add_trace(go.Scatter(x=global_rate['Date'],
                        y=global_rate['Global_Deaths_rate_%'],
                        mode='lines+markers',
                        name='Global Deaths rate %',
                        line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=global_rate['Date'],
                        y=global_rate['Global_Recovered_rate_%'],
                        mode='lines+markers',
                        name='Global Recovered rate %',
                        line=dict(color='green', width=2)))

fig.show()


# **Interactive Graphs - Q3:**
# 
# **Growth in new Confirmed and Deaths cases per day in the world**

# In[ ]:


#Daily growth confirmed cases
increase_daily = global_rate.loc[:,['Date', 'Increase_New_Cases_by_Day']]

fig_increase_daily = go.Figure()
fig_increase_daily.update_layout(
    title_text='Increase in new confirmed cases per day in the world',
    height=800, 
    width=1200, 
    xaxis_title='Period Date', 
    yaxis_title='New Confirmed Cases',
    template='plotly_dark')

fig_increase_daily.add_trace(go.Bar(
    x=increase_daily["Date"],
    y=increase_daily["Increase_New_Cases_by_Day"],
    marker_color='yellow',
    marker_line_color='rgb(8,48,107)',
    marker_line_width=1, 
    opacity=0.7)
             )

fig_increase_daily.show()


###


#Daily growth deaths cases
increase_daily_deaths = global_rate.loc[:,['Date', 'Increase_Deaths_Cases_per_day']]

fig_increase_daily_deaths = go.Figure()
fig_increase_daily_deaths.update_layout(
    title_text='Increase in new deaths cases per day in the world',
    height=800, 
    width=1200, 
    xaxis_title='Period Date', 
    yaxis_title='New Deaths Cases',
    template='plotly_dark')

fig_increase_daily_deaths.add_trace(go.Bar(
    x=increase_daily_deaths["Date"],
    y=increase_daily_deaths["Increase_Deaths_Cases_per_day"],
    marker_color='red',
    marker_line_color='rgb(8,48,107)',
    marker_line_width=1, 
    opacity=0.7)
             )

fig_increase_daily_deaths.show()


# **Table 01 - Q4 & Q5:**
# 
# **Total numbers confirmed, deaths, recovered, active cases and lethality rate by Country (list - 60 countries)**

# In[ ]:


global_cases = covid19confirmed 
global_cases = global_cases[['Country/Region', last_update]]
global_cases = global_cases.groupby('Country/Region').sum().sort_values(by = last_update,ascending = False)
global_cases['Deaths'] = covid19deaths[['Country/Region', last_update]].groupby('Country/Region').sum().sort_values(by = last_update, ascending = False)
global_cases['Recovered'] = covid19recovered[['Country/Region', last_update]].groupby('Country/Region').sum().sort_values(by = last_update, ascending = False)
global_cases['Active'] = global_cases[last_update] - global_cases['Recovered'] - global_cases['Deaths']
global_cases['Lethality Rate %'] = ((global_cases['Deaths'])/(global_cases[last_update])*100).round(2)
global_cases = global_cases.rename(columns = {last_update: 'Confirmed'})

#truncating in 60 countries
global_cases.head(60)


# # Predictions in the World using Machine Learning Algorithm - Prophet - procedure to forecasting time series data

# In[ ]:


prediction = covid19.copy()
prediction = prediction.groupby(['Date', 'Country/Region']).agg({'Confirmed':['sum'], 'Deaths':['sum'], 'Recovered':['sum']})
prediction.columns = ['Confirmed', 'Deaths', 'Recovered']
prediction = prediction.reset_index()
prediction = prediction[prediction.Confirmed!=0]
prediction = prediction[prediction.Deaths!=0]

#Prevent division by zero
def ifNull(d):
    temp=1
    if d!=0:
        temp=d
    return temp

prediction['mortality_rate'] = prediction.apply(lambda row: ((row.Deaths+1)/ifNull((row.Confirmed)))*100, axis=1)


# In[ ]:


floorVar = 0
worldPop = 25000000

#Modelling confirmed cases 
confirmed_train_dataset = pd.DataFrame(covid19.groupby('Date')['Confirmed'].sum().reset_index()).rename(columns={'Date': 'ds', 'Confirmed': 'y'})
confirmed_train_dataset['floor'] = floorVar
confirmed_train_dataset['cap'] = worldPop

#Modelling deaths cases
deaths_train_dataset = pd.DataFrame(covid19.groupby('Date')['Deaths'].sum().reset_index()).rename(columns={'Date': 'ds', 'Deaths': 'y'})
deaths_train_dataset['floor'] = 0
deaths_train_dataset['cap'] = 1000000

#Modelling mortality rate
mortality_train_dataset = pd.DataFrame(prediction.groupby('Date')['mortality_rate'].mean().reset_index()).rename(columns={'Date': 'ds', 'mortality_rate': 'y'})


# In[ ]:


#Dataframes 

#Confirmed model
m = Prophet(
    growth="logistic",
    interval_width=0.98,
    yearly_seasonality=False,
    weekly_seasonality=False,
    seasonality_mode='additive')

m.fit(confirmed_train_dataset)
future = m.make_future_dataframe(periods=50)
future['cap'] = worldPop
future['floor'] = floorVar
confirmed_forecast = m.predict(future)

#Mortality rate model
m_mortality = Prophet()
m_mortality.fit(mortality_train_dataset)
mortality_future = m_mortality.make_future_dataframe(periods=31)
mortality_forecast = m_mortality.predict(mortality_future)

#Deaths model
m2 = Prophet(
    growth="logistic",
    interval_width=0.95)
m2.fit(deaths_train_dataset)
future2 = m2.make_future_dataframe(periods=31)
future2['cap'] = 1000000
future2['floor'] = 0
deaths_forecast = m2.predict(future2)


# **Interactive Graph - Q6:** 
# 
# **Forecast of the covid-19 progress in the world to Confirmed cases**

# In[ ]:


fig_confirmed = plot_plotly(m, confirmed_forecast)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.10,
                       xanchor='left', yanchor='bottom',
                       text='Predictions to Confirmed cases in the World',
                       font=dict(family='Arial',
                                size=25,
                                color='rgb(37,37,37)'),
                       showarrow=False))
fig_confirmed.update_layout(annotations=annotations)
fig_confirmed


# **Interactive Graph - Q6:**
# 
# **Forecast of the covid-19 progress in the world to Deaths cases**

# In[ ]:


fig_deaths = plot_plotly(m2, deaths_forecast)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.10,
                       xanchor='left', yanchor='bottom',
                       text='Predictions to Deaths cases in the World',
                       font=dict(family='Arial',
                                size=25,
                                color='rgb(37,37,37)'),
                       showarrow=False))
fig_deaths.update_layout(annotations=annotations)
fig_deaths


# **Interactive Graph - Q7:**
# 
# **Forecast to lethality rate in the World**

# In[ ]:


fig_lethality = plot_plotly(m_mortality, mortality_forecast)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.10,
                       xanchor='left', yanchor='bottom',
                       text='Predictions to Lethality rate in the World (%)',
                       font=dict(family='Arial',
                                size=25,
                                color='rgb(37,37,37)'),
                       showarrow=False))
fig_lethality.update_layout(annotations=annotations)
fig_lethality


# # Brazil:
# 
# # Analysis of the advancement of the covid-19 in Brazil

# In[ ]:


Brazil_cases = covid19.copy()
Brazil_cases = covid19.loc[covid19['Country/Region']=='Brazil']
Brazil_cases = Brazil_cases.groupby(['Date', 'Country/Region']).agg({'Confirmed':['sum'], 'Deaths':['sum'], 'Recovered':['sum']}).sort_values('Date', ascending=False)
Brazil_cases.columns = ['Confirmed', 'Deaths', 'Recovered']
Brazil_cases = Brazil_cases.reset_index()
Brazil_cases['Confirmed_New_Daily_Cases'] = Brazil_cases['Confirmed'].diff().shift(-1)
Brazil_cases['Deaths_New_Daily_Cases'] = Brazil_cases['Deaths'].diff().shift(-1)
Brazil_cases['Recovered_New_Daily_Cases'] = Brazil_cases['Recovered'].diff().shift(-1)
Brazil_cases_confirmed = Brazil_cases[Brazil_cases['Confirmed']!=0]


# **Interactive Graph - Q8:**
# 
# **Total confirmed, deaths and recoveries cases**

# In[ ]:


fig = go.Figure()
fig.update_layout(title_text='Confirmed, Deaths and Recoveries cases in Brazil',
                        xaxis_title='Period Date', 
                        yaxis_title='Cases', 
                        template='plotly_dark')

fig.add_trace(go.Scatter(x=Brazil_cases_confirmed['Date'],
                        y=Brazil_cases_confirmed['Confirmed'],
                        mode='lines+markers',
                        name='Brazil Confirmed Cases',
                        line=dict(color='yellow', width=2)))

fig.add_trace(go.Scatter(x=Brazil_cases_confirmed['Date'],
                        y=Brazil_cases_confirmed['Deaths'],
                        mode='lines+markers',
                        name='Brazil Deaths Cases',
                        line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=Brazil_cases_confirmed['Date'],
                        y=Brazil_cases_confirmed['Recovered'],
                        mode='lines+markers',
                        name='Brazil Recovered Cases',
                        line=dict(color='green', width=2)))

fig.show()


# **Interactive Graph - Q9:**
# 
# **Growth rate of confirmed, deaths and recoveries cases**

# In[ ]:


Brazil_cases_rate = covid19.copy()
Brazil_cases_rate = covid19.loc[covid19['Country/Region']=='Brazil']
Brazil_cases_rate = Brazil_cases.groupby(['Date', 'Country/Region']).agg({'Confirmed':['sum'], 'Deaths':['sum'], 'Recovered':['sum']})
Brazil_cases_rate.columns = ['Confirmed', 'Deaths', 'Recovered']
Brazil_cases_rate = Brazil_cases_rate.reset_index()
Brazil_cases_rate['Confirmed_New_Daily_Cases'] = Brazil_cases_rate['Confirmed'].diff().shift(+1)
Brazil_cases_rate['Deaths_New_Daily_Cases'] = Brazil_cases_rate['Deaths'].diff().shift(+1)
Brazil_cases_rate = Brazil_cases_rate[Brazil_cases_rate.Confirmed!=0]
Brazil_cases_rate = Brazil_cases_rate[Brazil_cases_rate.Deaths!=0]

#Calculating rate
#Lambda function
Brazil_cases_rate['Brazil_Deaths_rate_%'] = Brazil_cases_rate.apply(lambda row: ((row.Deaths)/(row.Confirmed))*100, axis=1).round(2)
Brazil_cases_rate['Brazil_Recovered_rate_%'] = Brazil_cases_rate.apply(lambda row: ((row.Recovered)/(row.Confirmed))*100, axis=1).round(2)
Brazil_cases_rate['Brazil_Growth_rate_%'] = Brazil_cases_rate.apply(lambda row: row.Confirmed_New_Daily_Cases/row.Confirmed*100, axis=1).round(2)
Brazil_cases_rate['Brazil_Growth_rate_%'] = Brazil_cases_rate['Brazil_Growth_rate_%'].shift(+1)
Brazil_cases_rate['Brazil_Growth_Deaths_rate_%'] = Brazil_cases_rate.apply(lambda row: row.Deaths_New_Daily_Cases/row.Confirmed*100, axis=1).round(2)
Brazil_cases_rate['Brazil_Growth_Deaths_rate_%'] = Brazil_cases_rate['Brazil_Growth_Deaths_rate_%'].shift(+1)

fig = go.Figure()
fig.update_layout(title_text='Growth rate of confirmed, deaths and recoveries cases in Brazil (%):',
                        xaxis_title='Period Date', 
                        yaxis_title='Rate', 
                        template='plotly_dark')

fig.add_trace(go.Scatter(x=Brazil_cases_rate['Date'],
                        y=Brazil_cases_rate['Brazil_Growth_rate_%'],
                        mode='lines+markers',
                        name='Growth Confirmed Cases Rate in Brazil %',
                        line=dict(color='yellow', width=2)))

fig.add_trace(go.Scatter(x=Brazil_cases_rate['Date'],
                        y=Brazil_cases_rate['Brazil_Deaths_rate_%'],
                        mode='lines+markers',
                        name='Deaths Rate in Brazil %',
                        line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=Brazil_cases_rate['Date'],
                        y=Brazil_cases_rate['Brazil_Recovered_rate_%'],
                        mode='lines+markers',
                        name='Recovered Rate in Brazil %',
                        line=dict(color='green', width=2)))

fig.show()


# **Interactive Graphs - Q10:**
# 
# **Daily growth in the number of confirmed and deaths cases in Brazil**

# In[ ]:


#Daily increase confirmed cases
increase_daily_brazil = Brazil_cases_rate.loc[:,['Date', 'Confirmed_New_Daily_Cases']]

fig_increase_daily_brazil = go.Figure()
fig_increase_daily_brazil.update_layout(
    title_text='Increase in new confirmed cases per day in Brazil',
    height=800, 
    width=1200, 
    xaxis_title='Period Date', 
    yaxis_title='New Confirmed Cases',
    )

fig_increase_daily_brazil.add_trace(go.Bar(
    x=increase_daily_brazil["Date"],
    y=increase_daily_brazil["Confirmed_New_Daily_Cases"],
    marker_color='yellow',
    marker_line_color='rgb(8,48,107)',
    marker_line_width=1, 
    opacity=0.7)
             )

fig_increase_daily_brazil.show()


###


#Daily increase deaths cases
increase_daily_deaths_brazil = Brazil_cases_rate.loc[:,['Date', 'Deaths_New_Daily_Cases']]

fig_increase_daily_deaths_brazil = go.Figure()
fig_increase_daily_deaths_brazil.update_layout(
    title_text='Increase in new deaths cases per day in Brazil',
    height=800, 
    width=1200, 
    xaxis_title='Period Date', 
    yaxis_title='New Deaths Cases',
    )

fig_increase_daily_deaths_brazil.add_trace(go.Bar(
    x=increase_daily_deaths_brazil["Date"],
    y=increase_daily_deaths_brazil["Deaths_New_Daily_Cases"],
    marker_color='red',
    marker_line_color='rgb(8,48,107)',
    marker_line_width=1, 
    opacity=0.7)
             )

fig_increase_daily_deaths_brazil.show()


# # Province/Region of Brazil
# 
# **Interactive Graphs and Table 02 - Q11 & Q12:**
# 
# **Confirmed and Deaths cases by region**
# 
# **Lethality rate by region**

# In[ ]:


Brazil_cases_region = covid19Brazil.loc[:,['region', 'cases', 'deaths','date']].groupby(['region', 'date']).sum().reset_index().sort_values(['cases', 'date'], ascending=False)
Brazil_cases_region = Brazil_cases_region.drop_duplicates(subset = ['region'])
Brazil_cases_region = Brazil_cases_region.set_index('date')
Brazil_cases_region['Lethality Rate %'] = ((Brazil_cases_region['deaths'])/(Brazil_cases_region['cases'])*100).round(2)
Brazil_cases_region_rename = Brazil_cases_region.rename(columns = {'region': 'Region', 'cases' : 'Confirmed', 'deaths' : 'Deaths'})

#Confirmed cases
fig = go.Figure()
fig.update_layout(
    title_text='Confirmed cases by region to date',
    height=400, 
    width=500, 
    xaxis_title='Regions', 
    yaxis_title='Confirmed Cases')

fig.add_trace(go.Bar(
    x=Brazil_cases_region["region"],
    y=Brazil_cases_region["cases"],
    name='Confirmed cases',
    marker_color='darkcyan',
    marker_line_color='rgb(8,48,107)',
    marker_line_width=2, 
    opacity=0.7)
             )

fig.show()


###


#Deaths cases
fig2 = go.Figure()
fig2.update_layout(
    title_text='Deaths cases by region to date',
    height=400, 
    width=500, 
    xaxis_title='Regions', 
    yaxis_title='Deaths')

fig2.add_trace(go.Bar(
    x=Brazil_cases_region["region"],
    y=Brazil_cases_region["deaths"],
    name='Deaths',
    marker_color='red',
    marker_line_color='rgb(8,48,107)',
    marker_line_width=2, 
    opacity=0.7)
             )

fig2.show()


###


#Lethality rate
fig3 = go.Figure()
fig3.update_layout(
    title_text='Lethality rate by region to date',
    height=400, 
    width=500, 
    xaxis_title='Regions', 
    yaxis_title='Lethality Rate %')

fig3.add_trace(go.Bar(
    x=Brazil_cases_region["region"],
    y=Brazil_cases_region["Lethality Rate %"],
    name='Lethality Rate',
    marker_color='orangered',
    marker_line_color='rgb(8,48,107)',
    marker_line_width=2, 
    opacity=0.7)
             )

fig3.show()


Brazil_cases_region_rename


# # States of Brazil
# 
# **Interactive Graphs and Table 03 - Q13 & Q14:**
# 
# **Confirmed cases by state**

# In[ ]:


Brazil_cases_state = covid19Brazil.groupby(['state', 'date']).sum().reset_index().sort_values(['cases', 'deaths', 'date'], ascending=False)
Brazil_cases_state = Brazil_cases_state.drop_duplicates(subset = ['state'])
Brazil_cases_state = Brazil_cases_state.set_index('date')
Brazil_cases_state['Lethality Rate %'] = ((Brazil_cases_state['deaths'])/(Brazil_cases_state['cases'])*100).round(2)
Brazil_cases_state_rename = Brazil_cases_state.rename(columns = {'state': 'State', 'cases' : 'Confirmed', 'deaths' : 'Deaths'})

fig_brazil_cases_state = go.Figure()
fig_brazil_cases_state.update_layout(
    title_text='States of Brazil - Confirmed cases to date',
    height=800, 
    width=1500, 
    xaxis_title='Cases',
    yaxis_title='States')

fig_brazil_cases_state.add_trace(go.Bar(
    x=Brazil_cases_state["cases"],
    y=Brazil_cases_state["state"],
    orientation='h',
    marker_color='darkcyan',
    marker_line_color='rgb(8,48,107)',
    marker_line_width=2,
    opacity=0.7))

fig_brazil_cases_state.show()


# **Deaths cases by state**

# In[ ]:


Brazil_cases_state_deaths = covid19Brazil.groupby(['state', 'date']).sum().reset_index().sort_values(['deaths', 'date'], ascending=False)
Brazil_cases_state_deaths = Brazil_cases_state_deaths.drop_duplicates(subset = ['state'])
Brazil_cases_state_deaths = Brazil_cases_state_deaths.set_index('date')

fig_brazil_cases_state_deaths = go.Figure()
fig_brazil_cases_state_deaths.update_layout(
    title_text='States of Brazil - Deaths cases to date',
    height=800, 
    width=1500, 
    xaxis_title='Cases',
    yaxis_title='States')

fig_brazil_cases_state_deaths.add_trace(go.Bar(
    x=Brazil_cases_state_deaths["deaths"],
    y=Brazil_cases_state_deaths["state"],
    orientation='h',
    marker_color='red',
    marker_line_color='rgb(8,48,107)',
    marker_line_width=2,
    opacity=0.7))

fig_brazil_cases_state_deaths


# **Lethality rate in each state**

# In[ ]:


#Table 03
Brazil_cases_state_rename


# # Predictions in Brazil using Machine Learning Algorithm - Prophet - procedure for forecasting time series data

# In[ ]:


prediction_brazil = covid19Brazil.copy()
prediction_brazil = prediction_brazil.groupby(['date']).agg({'deaths':['sum'], 'cases':['sum']})
prediction_brazil.columns = ['deaths', 'cases']
prediction_brazil = prediction_brazil.reset_index()
prediction_brazil = prediction_brazil[prediction_brazil.deaths!=0]
prediction_brazil = prediction_brazil[prediction_brazil.cases!=0]

#Prevent division by zero
def ifNull(d):
    temp=1
    if d!=0:
        temp=d
    return temp

prediction_brazil['Brazil_mortality_rate_%'] = prediction_brazil.apply(lambda row: ((row.deaths+1)/ifNull((row.cases)))*100, axis=1)


# In[ ]:


floorVar=0
BrazilPop=5000000

#Modelling confirmed cases
brazil_confirmed_train_dataset = pd.DataFrame(covid19Brazil.groupby('date')['cases'].sum().reset_index()).rename(columns={'date':'ds', 'cases':'y'})
brazil_confirmed_train_dataset['floor'] = floorVar
brazil_confirmed_train_dataset['cap'] = BrazilPop

#Modelling mortality rate
brazil_mortality_train_dataset = pd.DataFrame(prediction_brazil.groupby('date')['Brazil_mortality_rate_%'].mean().reset_index()).rename(columns={'date':'ds', 'Brazil_mortality_rate_%':'y'})

#Modelling deaths cases
brazil_deaths_train_dataset = pd.DataFrame(covid19Brazil.groupby('date')['deaths'].sum().reset_index()).rename(columns={'date': 'ds', 'deaths': 'y'})
brazil_deaths_train_dataset['floor'] = 0
brazil_deaths_train_dataset['cap'] = 150000


# In[ ]:


#Dataframes 

#Confirmed model
m_brazil = Prophet(
    growth="logistic",
    interval_width=0.98,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=True,
    seasonality_mode='additive'
    )

m_brazil.fit(brazil_confirmed_train_dataset)
future_brazil = m_brazil.make_future_dataframe(periods=50)
future_brazil['cap']=BrazilPop
future_brazil['floor']=floorVar
forecast_brazil_confirmed = m_brazil.predict(future_brazil)

#Mortality rate model
m_brazil_mortality = Prophet()
m_brazil_mortality.fit(brazil_mortality_train_dataset)
future_brazil_mortality = m_brazil_mortality.make_future_dataframe(periods=31)
forecast_brazil_mortality = m_brazil_mortality.predict(future_brazil_mortality)

#Deaths model
m2_brazil = Prophet(
    interval_width=0.95,
    growth="logistic"
    )

m2_brazil.fit(brazil_deaths_train_dataset)
future2_brazil = m2_brazil.make_future_dataframe(periods=31)
future2_brazil['cap']=150000
future2_brazil['floor']=0
forecast_brazil_deaths = m2_brazil.predict(future2_brazil)


# **Interactive Graph - Q15:**
# 
# **Forecast of the disease's progress in Brazil for Confirmed cases**

# In[ ]:


fig_confirmed_brazil = plot_plotly(m_brazil, forecast_brazil_confirmed)
annotations = []
annotations.append(dict(
                        xref='paper',yref='paper', x=0.0, y=1.10,
                        xanchor='left',yanchor='bottom',
                        text='Predictions to Confirmed cases in Brazil',
                        font=dict(family='Arial',
                                 size=25,
                                 color='rgb(37,37,37)'),
                                showarrow=False))
fig_confirmed_brazil.update_layout(annotations=annotations)
fig_confirmed_brazil


# **Interactive Graph - Q15:**
# 
# **Forecast of the disease's progress in Brazil for Deaths cases**

# In[ ]:


fig_deaths_brazil = plot_plotly(m2_brazil, forecast_brazil_deaths) 
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.10,
                              xanchor='left', yanchor='bottom',
                              text='Predictions to Deaths cases in Brazil',
                              font=dict(family='Arial',
                                        size=25,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig_deaths_brazil.update_layout(annotations=annotations)
fig_deaths_brazil


# **Interactive Graph - Q16:**
# 
# **Forecast to Lethality rate in Brazil**

# In[ ]:


fig_lethality_brazil = plot_plotly(m_brazil_mortality, forecast_brazil_mortality)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.10,
                              xanchor='left', yanchor='bottom',
                              text='Predictions to Lethality rate in Brazil (%)',
                              font=dict(family='Arial',
                                        size=25,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig_lethality_brazil.update_layout(annotations=annotations)
fig_lethality_brazil


# # State of Pernambuco (where i live) - Analysis and Predictions
# 
# **Interactive Graph - Q17:**
# 
# **Confirmed and Deaths cases**

# In[ ]:


cases_pernambuco = covid19Brazil.copy()
cases_pernambuco = covid19Brazil.loc[covid19Brazil['state']=='PE']
cases_pernambuco = cases_pernambuco.groupby(['date']).agg({'cases':['sum'], 'deaths':['sum']}).sort_values('date', ascending=False)
cases_pernambuco.columns = ['cases', 'deaths']
cases_pernambuco = cases_pernambuco.reset_index()
cases_pernambuco = cases_pernambuco[cases_pernambuco['cases']!=0]

fig = go.Figure()
fig.update_layout(
    title_text='State of Pernambuco - Confirmed and Deaths cases',
    xaxis_title='Period Date',
    yaxis_title='Cases',
    template='seaborn',
    width=1200,
    height=600)

fig.add_trace(go.Scatter(
    x=cases_pernambuco['date'],
    y=cases_pernambuco['cases'],
    mode='lines+markers',
    name='Confirmed cases',
    line=dict(color='darkcyan', width=2)))

fig.add_trace(go.Scatter(
    x=cases_pernambuco['date'],
    y=cases_pernambuco['deaths'],
    mode='lines+markers',
    name='Deaths cases',
    line=dict(color='red', width=2)))


# **Interactive Graph - Q18:**
# 
# **Confirmed and Deaths cases - Rate %**

# In[ ]:


cases_pernambuco_rate = covid19Brazil.copy()
cases_pernambuco_rate = covid19Brazil.loc[covid19Brazil['state']=='PE']
cases_pernambuco_rate = cases_pernambuco_rate.groupby(['date']).agg({'deaths':['sum'], 'cases':['sum']})
cases_pernambuco_rate.columns = ['deaths', 'cases']
cases_pernambuco_rate = cases_pernambuco_rate.reset_index()
cases_pernambuco_rate['Confirmed_New_Daily_Cases_in_Pernambuco'] = cases_pernambuco_rate['cases'].diff().shift(0)
cases_pernambuco_rate['Deaths_New_Daily_Cases_in_Pernambuco'] = cases_pernambuco_rate['deaths'].diff().shift(0)
cases_pernambuco_rate = cases_pernambuco_rate[cases_pernambuco_rate.cases!=0]
cases_pernambuco_rate = cases_pernambuco_rate[cases_pernambuco_rate.deaths!=0]

#Calculating rate
#Lambda function
cases_pernambuco_rate['Pernambuco_Deaths_rate_%'] = cases_pernambuco_rate.apply(lambda row: ((row.deaths)/(row.cases))*100, axis=1).round(2)
cases_pernambuco_rate['Pernambuco_Growth_rate_%'] = cases_pernambuco_rate.apply(lambda row: row.Confirmed_New_Daily_Cases_in_Pernambuco/row.cases*100, axis=1).round(2)
cases_pernambuco_rate['Pernambuco_Growth_rate_%'] = cases_pernambuco_rate['Pernambuco_Growth_rate_%'].shift(+1)

fig = go.Figure()
fig.update_layout(
    title_text='State of Pernambuco - Growth rate of Confirmed and Deaths (%)', 
    xaxis_title='Period Date', 
    yaxis_title='Rate', 
    template='seaborn', 
    width=1200, 
    height=600)

fig.add_trace(go.Scatter(
    x=cases_pernambuco_rate['date'], 
    y=cases_pernambuco_rate['Pernambuco_Growth_rate_%'],
    mode='lines+markers',
    name='Confirmed rate %',
    line=dict(color='darkcyan', width=2)))

fig.add_trace(go.Scatter(
    x=cases_pernambuco_rate['date'], 
    y=cases_pernambuco_rate['Pernambuco_Deaths_rate_%'],
    mode='lines+markers',
    name='Deaths rate %',
    line=dict(color='red', width=2)))


# **Interactive Graphs - Q19:**
# 
# **Daily growth in the number of confirmed and deaths cases in Pernambuco**

# In[ ]:


#Daily increase confirmed cases
increase_daily_pernambuco = cases_pernambuco_rate.loc[:,['date', 'Confirmed_New_Daily_Cases_in_Pernambuco']]

fig_increase_daily_pernambuco = go.Figure()
fig_increase_daily_pernambuco.update_layout(
    title_text='Increase in new confirmed cases per day in Pernambuco',
    height=800, 
    width=1200, 
    xaxis_title='Period Date', 
    yaxis_title='New Confirmed Cases',
    )

fig_increase_daily_pernambuco.add_trace(go.Bar(
    x=increase_daily_pernambuco["date"],
    y=increase_daily_pernambuco["Confirmed_New_Daily_Cases_in_Pernambuco"],
    marker_color='darkcyan',
    marker_line_color='rgb(8,48,107)',
    marker_line_width=1, 
    opacity=0.7)
             )

fig_increase_daily_pernambuco.show()


###


#Daily increase deaths cases
increase_daily_deaths_pernambuco = cases_pernambuco_rate.loc[:,['date', 'Deaths_New_Daily_Cases_in_Pernambuco']]

fig_increase_daily_deaths_pernambuco = go.Figure()
fig_increase_daily_deaths_pernambuco.update_layout(
    title_text='Increase in new deaths cases per day in Pernambuco',
    height=800, 
    width=1200, 
    xaxis_title='Period Date', 
    yaxis_title='New Deaths Cases',
    )

fig_increase_daily_deaths_pernambuco.add_trace(go.Bar(
    x=increase_daily_deaths_pernambuco["date"],
    y=increase_daily_deaths_pernambuco["Deaths_New_Daily_Cases_in_Pernambuco"],
    marker_color='red',
    marker_line_color='rgb(8,48,107)',
    marker_line_width=1, 
    opacity=0.7)
             )

fig_increase_daily_deaths_pernambuco.show()


# # Predictions in Pernambuco using Machine Learning Algorithm - Prophet - procedure for forecasting time series data

# In[ ]:


prediction_state_pernambuco = covid19Brazil.copy()
prediction_state_pernambuco = covid19Brazil.loc[covid19Brazil['state']=='PE']
prediction_state_pernambuco = prediction_state_pernambuco.groupby(['date']).agg({'deaths':['sum'], 'cases':['sum']})
prediction_state_pernambuco.columns = ['deaths', 'cases']
prediction_state_pernambuco = prediction_state_pernambuco.reset_index()
prediction_state_pernambuco = prediction_state_pernambuco[prediction_state_pernambuco.deaths!=0]
prediction_state_pernambuco = prediction_state_pernambuco[prediction_state_pernambuco.cases!=0]

#Prevent division by zero
def ifNull(d):
    temp=1
    if d!=0:
        temp=d
    return temp

prediction_state_pernambuco['Pernambuco_mortality_rate_%'] = prediction_state_pernambuco.apply(lambda row: ((row.deaths+1)/ifNull((row.cases)))*100, axis=1)


# In[ ]:


floorVar=0
PernambucoPop=150000

#Modelling confirmed cases
pernambuco_confirmed_train_dataset = pd.DataFrame(covid19Brazil.loc[covid19Brazil['state']=='PE'].groupby('date')['cases'].sum().reset_index()).rename(columns={'date':'ds', 'cases':'y'})
pernambuco_confirmed_train_dataset['floor'] = floorVar
pernambuco_confirmed_train_dataset['cap'] = PernambucoPop

#Modelling mortality rate
pernambuco_mortality_train_dataset = pd.DataFrame(prediction_state_pernambuco.groupby('date')['Pernambuco_mortality_rate_%'].mean().reset_index()).rename(columns={'date':'ds', 'Pernambuco_mortality_rate_%':'y'})

#Modelling deaths cases
pernambuco_deaths_train_dataset = pd.DataFrame(covid19Brazil.loc[covid19Brazil['state']=='PE'].groupby('date')['deaths'].sum().reset_index()).rename(columns={'date':'ds', 'deaths':'y'})
pernambuco_deaths_train_dataset['floor'] = 0
pernambuco_deaths_train_dataset['cap'] = 10000


# In[ ]:


#Dataframes 

#Confirmed model
m_pernambuco = Prophet(
    growth="logistic",
    interval_width=0.98,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=True,
    seasonality_mode='additive'
    )

m_pernambuco.fit(pernambuco_confirmed_train_dataset)
future_pernambuco = m_pernambuco.make_future_dataframe(periods=50)
future_pernambuco['cap']=PernambucoPop
future_pernambuco['floor']=floorVar
forecast_pernambuco_confirmed = m_pernambuco.predict(future_pernambuco)

#Mortality rate model
m_pernambuco_mortality = Prophet()
m_pernambuco_mortality.fit(pernambuco_mortality_train_dataset)
future_pernambuco_mortality = m_pernambuco_mortality.make_future_dataframe(periods=31)
forecast_pernambuco_mortality = m_pernambuco_mortality.predict(future_pernambuco_mortality)

#Deaths model
m2_pernambuco = Prophet(
    interval_width=0.95,
    growth="logistic"
    )

m2_pernambuco.fit(pernambuco_deaths_train_dataset)
future2_pernambuco = m2_pernambuco.make_future_dataframe(periods=31)
future2_pernambuco['cap']=10000
future2_pernambuco['floor']=0
forecast_pernambuco_deaths = m2_pernambuco.predict(future2_pernambuco)


# **Interactive Graph - Q20:**
# 
# **Forecast of the disease's progress in Pernambuco for Confirmed cases**

# In[ ]:


fig_confirmed_pernambuco = plot_plotly(m_pernambuco, forecast_pernambuco_confirmed)
annotations = []
annotations.append(dict(
                        xref='paper',yref='paper', x=0.0, y=1.10,
                        xanchor='left',yanchor='bottom',
                        text='Predictions to Confirmed cases in Pernambuco',
                        font=dict(family='Arial',
                                 size=25,
                                 color='rgb(37,37,37)'),
                                showarrow=False))
fig_confirmed_pernambuco.update_layout(annotations=annotations)
fig_confirmed_pernambuco


# **Interactive Graph - Q20:**
# 
# **Forecast of the disease's progress in Pernambuco for Deaths cases**

# In[ ]:


fig_deaths_pernambuco = plot_plotly(m2_pernambuco, forecast_pernambuco_deaths) 
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.10,
                              xanchor='left', yanchor='bottom',
                              text='Predictions to Deaths cases in Pernambuco',
                              font=dict(family='Arial',
                                        size=25,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig_deaths_pernambuco.update_layout(annotations=annotations)
fig_deaths_pernambuco


# **Interactive Graph - Q21:**
# 
# **Forecast to Lethality rate in Pernambuco**

# In[ ]:


fig_lethality_pernambuco = plot_plotly(m_pernambuco_mortality, forecast_pernambuco_mortality)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.10,
                              xanchor='left', yanchor='bottom',
                              text='Predictions to Lethality rate in Pernambuco (%)',
                              font=dict(family='Arial',
                                        size=25,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig_lethality_pernambuco.update_layout(annotations=annotations)
fig_lethality_pernambuco


# **Thank you for reading this kernel. It is the 1st release version (May, 5, 2020).**
# 
# **Your feedback will be really appreciated.**
# 
# **If you think that this analysis was useful please give a upvote, i like so much!**
# 
# **To be continued to updates and more versions.**
# 
# *Best regards,*
# 
# *Nilton Coura.*
# 
# *     contact: niltontac@gmail.com
# 
# *     https://github.com/niltontac
# 
# *     https://www.linkedin.com/in/niltontac/

# **Last Updates:**
# 
# *May, 7, 2020:*
# * Adjustment in Table 01
# * Adjustment in Death models (train)
# * Implementation of the Lethality Rate by region and state of Brazil
# 
# *May, 25, 2020:*
# * Implementation of the daily growth of number confirmed and deaths cases in Pernambuco

# Some preventions and symptoms:
# 
# ![](https://tmgcompaniesllc.com/wp-content/uploads/2020/03/140972119_m-1-800x457.jpg)
