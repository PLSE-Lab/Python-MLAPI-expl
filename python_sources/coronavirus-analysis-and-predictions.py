#!/usr/bin/env python
# coding: utf-8

# # Analysis of Coronavirus Data
# 
# 

# In[ ]:


# Importing packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go 
import seaborn as sns
import plotly
import plotly.express as px
from fbprophet.plot import plot_plotly
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot_mpl
import plotly.offline as py
init_notebook_mode(connected=True)

plt.rcParams.update({'font.size': 14})
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates = ['ObservationDate','Last Update'])
data.shape


# Checking the latest 10 cases to see when the dataset was last updated. 

# In[ ]:


data.head(10)


# In[ ]:


data['Country/Region'].unique()


# In[ ]:


# data = data.drop_duplicates()
data = data.drop(['SNo', 'Last Update', 'Province/State'], axis=1)
data = data.rename(columns={'Country/Region': 'Country', 'ObservationDate':'Date'})
# To check null values
data.isnull().sum()


# In[ ]:


data.head(2)


# Note that in the dataset, a place may have reported data more than once per day. For more effective analysis, we convert the data into daily. If the data for the latest day is not available, we will fill it with previous available data.

# In[ ]:


#This creates a table that sums up every element in the Confirmed, Deaths, and recovered columns.
temp = data.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum()
#Reset index coverts the index series, in this case date, into an index value. 
temp = temp.reset_index()
temp = temp.sort_values('Date', ascending=False)
temp['Mortality Rate %']=temp['Deaths']/temp['Confirmed']*100
temp.head().style.background_gradient(cmap='PRGn')


# In[ ]:


#Confirmed ALL
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=temp['Date'], 
                         y=temp['Confirmed'],
                         mode='lines+markers',
                         name='Confirmed',
                         line=dict(color='Yellow', width=2)))
fig.show()


# ## Plotly visualizations

# ## Confirmed Visualisations

# In[ ]:


Top_3_affected = data.copy()
Screwed=['Iran','Italy','South Korea']#,'Mainland China']
Top_3_affected = Top_3_affected[Top_3_affected.Country.isin(Screwed)]
Top_3_affected = Top_3_affected.groupby(['Date', 'Country']).agg({'Confirmed': ['sum']})
Top_3_affected.columns = ['Confirmed Cases']
Top_3_affected = Top_3_affected.reset_index()
fig = px.line(Top_3_affected, x="Date", y="Confirmed Cases", color="Country",
              line_group="Country", hover_name="Country")
fig.update_layout(template='plotly_dark')
fig.show()


# Copying the dataframe into a new dataframe and deleting the columns for China, to get an image of what the status is for the coronavirus in the rest of the world. There has been a lot of talk about Chinese sources being unreliable. Thus modelling how the virus will progress based on Chinese data only can give a skewed result, and if the hypothesis is to be believed that the chinese government is lying about their numbers and artificially deflating them, then a predictive model will give a too low result and perhaps enact a false sense of security. Not providing an urgent response to an epidemic can be the thing that results in the epidemic progressing into a pandemic. 

# In[ ]:


grouped_multiple = data.groupby(['Date']).agg({'Deaths': ['sum'],'Recovered': ['sum'],'Confirmed': ['sum']})
grouped_multiple.columns = ['Deaths_ALL','Recovered_ALL','Confirmed_ALL']
grouped_multiple = grouped_multiple.reset_index()
grouped_multiple['Difference_world']=grouped_multiple['Confirmed_ALL'].diff().shift(-1)

grouped_multiple['Deaths_ALL_%'] = grouped_multiple.apply(lambda row: ((row.Deaths_ALL)/(row.Confirmed_ALL))*100 , axis=1)
grouped_multiple['Recovered_ALL_%'] = grouped_multiple.apply(lambda row: ((row.Recovered_ALL)/(row.Confirmed_ALL))*100 , axis=1)
grouped_multiple['World_growth_rate']=grouped_multiple.apply(lambda row: row.Difference_world/row.Confirmed_ALL*100, axis=1)


fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 
                         y=grouped_multiple['Deaths_ALL_%'],
                         mode='lines+markers',
                         name='Death rate',
                         line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 
                         y=grouped_multiple['Recovered_ALL_%'],
                         mode='lines+markers',
                         name='Recovery rate',
                         line=dict(color='Green', width=2)))

fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 
                         y=grouped_multiple['World_growth_rate'],
                         mode='lines+markers',
                         name='Growth rate confirmed',
                         line=dict(color='Yellow', width=2)))

fig.show()
grouped_multiple.tail()


# Calculating the percentage change in the growth rates. To see if the growth in total cases is accelerrating or decellerating. 

# Evolution in mortality rates over time by Country

# In[ ]:


mortality = data.copy()

mortality = mortality[mortality.Country.isin(Screwed)]
mortality = mortality.groupby(['Date', 'Country']).agg({'Deaths': ['sum'],'Recovered': ['sum'],'Confirmed': ['sum']})
mortality.columns = ['Deaths','Recovered','Confirmed']
mortality = mortality.reset_index()
mortality = mortality[mortality.Deaths != 0]
mortality = mortality[mortality.Confirmed != 0]

mortality['Mortality_Rate'] = mortality.apply(lambda row: ((row.Deaths+1)/(row.Confirmed+1))*100, axis=1)

#We filter out where mortality rate is above 10% 
d = mortality[mortality.Mortality_Rate < 10]

#We wannt only to plot countries with more than 100 confirmed cases, as the situation evovles, more countries will enter this list.
dd = d[d.Confirmed > 100]

fig = px.line(dd, x="Date", y="Mortality_Rate", color="Country",
              line_group="Country", hover_name="Country")
fig.update_layout(template='plotly_dark')
fig.show()

##add average



# ## Machine learning with facebook prophet

# Worldpop is the variable that holds the cap rate nessecary for fb prophet algoritm to work. Prophet requires columns to be labelled ds and y. For the logaritmic model a cap rate and a floor is nessecary. These are inserted into the pandas dataframe. We are using a constant cap rate. Right now its set at 500k.

# In[ ]:


floorVar=0.8
worldPop=25000

#Modelling total confirmed cases 
confirmed_training_dataset = pd.DataFrame(data.groupby('Date')['Confirmed'].sum().reset_index()).rename(columns={'Date': 'ds', 'Confirmed': 'y'})
#confirmed_training_dataset.insert(0,'floor',1)
confirmed_training_dataset['floor'] = confirmed_training_dataset.y*floorVar
confirmed_training_dataset['cap'] = confirmed_training_dataset.y+worldPop

#Modelling mortality rate
mortality_training_dataset = pd.DataFrame(mortality.groupby('Date')['Mortality_Rate'].mean().reset_index()).rename(columns={'Date': 'ds', 'Mortality_Rate': 'y'})

#Modelling deaths
death_training_dataset = pd.DataFrame(data.groupby('Date')['Deaths'].sum().reset_index()).rename(columns={'Date': 'ds', 'Deaths': 'y'})


# In[ ]:


# Total dataframe model 
m = Prophet(
    interval_width=0.90,
    changepoint_prior_scale=0.05,
    changepoint_range=0.9,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=True,
    seasonality_mode='additive'
    )

m.fit(confirmed_training_dataset)
future = m.make_future_dataframe(periods=61)
future['cap']=confirmed_training_dataset.y+worldPop
future['floor']=confirmed_training_dataset.y*floorVar
confirmed_forecast = m.predict(future)

# Mortality rate model
m_mortality = Prophet ()
m_mortality.fit(mortality_training_dataset)
mortality_future = m_mortality.make_future_dataframe(periods=31)
mortality_forecast = m_mortality.predict(mortality_future)

# Deaths model
m2 = Prophet(interval_width=0.95)
m2.fit(death_training_dataset)
future2 = m2.make_future_dataframe(periods=7)
death_forecast = m2.predict(future2)


# In[ ]:


fig = plot_plotly(m, confirmed_forecast)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Predictions for Total Confirmed cases',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig.update_layout(annotations=annotations)
fig


# ## Predictions for confirmed. 

# In[ ]:


fig = plot_plotly(m_mortality, mortality_forecast)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Predictions for mortality rate',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig.update_layout(annotations=annotations)
fig


# In[ ]:


fig_death = plot_plotly(m2, death_forecast)  
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Predictions for Deaths',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig_death.update_layout(annotations=annotations)
fig_death

