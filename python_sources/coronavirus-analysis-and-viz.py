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
import warnings

warnings.filterwarnings('ignore')

init_notebook_mode(connected=True)

plt.rcParams.update({'font.size': 14})
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates = ['ObservationDate','Last Update'])
country_data = pd.read_csv("../input/countries-of-the-world/countries of the world.csv")

data.shape


# Removing unessecary columns

# In[ ]:


# data = data.drop_duplicates()
data = data.drop(['SNo', 'Last Update'], axis=1)
data = data.rename(columns={'Country/Region': 'Country', 'ObservationDate':'Date'})
# To check null values
data.isnull().sum()


# Note that in the dataset, a place may have reported data more than once per day. For more effective analysis, we convert the data into daily. If the data for the latest day is not available, we will fill it with previous available data.

# In[ ]:


#This creates a table that sums up every element in the Confirmed, Deaths, and recovered columns.
temp = data.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum()
#Reset index coverts the index series, in this case date, into an index value. 
temp = temp.reset_index()
temp = temp.sort_values('Date', ascending=False)
temp.head().style.background_gradient(cmap='PRGn')


# In[ ]:


#Daily cases in countries 


# In[ ]:


#Confirmed ALL
fig = go.Figure()
fig.update_layout(template='seaborn',width=800, height=400)
fig.add_trace(go.Scatter(x=temp['Date'], 
                         y=temp['Confirmed'],
                         mode='lines+markers',
                         name='Confirmed',
                         line=dict(color='Yellow', width=2)))
fig.add_trace(go.Scatter(x=temp['Date'], 
                         y=temp['Deaths'],
                         mode='lines+markers',
                         name='Deaths',
                         line=dict(color='Red', width=2)))
fig.add_trace(go.Scatter(x=temp['Date'], 
                         y=temp['Recovered'],
                         mode='lines+markers',
                         name='Recovered',
                         line=dict(color='Green', width=2)))
fig.show()


# ## Plotly visualizations

# ## Confirmed Visualisations

# New daily cases

# In[ ]:


i=1
f = plt.figure(figsize=(10,15))
countries = ['Denmark','Sweden','Norway','Finland','Iceland']
for country in countries :
    data2 = data.copy()
    data2 = data.loc[data['Country'] == country]
    data2 = data2.groupby(['Date', 'Country']).agg({'Confirmed': ['sum']})
    data2.columns = ['Confirmed']
    data2 = data2.reset_index()
    data2['New cases']=data2['Confirmed'].diff()
    
    ax = f.add_subplot(5,1,i)
    ax.bar(data2['Date'],data2['New cases']);
    ax.set_title(country)
    ax.set_ylabel('Daily confirmed cases')
    if i != len(countries) : 
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.xticks(rotation=50)
    i=i+1





# In[ ]:


china_vs_rest = data.copy()

china_vs_rest = china_vs_rest.groupby(['Date', 'Country']).agg({'Confirmed': ['sum']})
china_vs_rest.columns = ['Confirmed ALL']
china_vs_rest = china_vs_rest.reset_index()
fig = px.line(china_vs_rest, x="Date", y="Confirmed ALL", color="Country",
              line_group="Country", hover_name="Country")
fig.update_layout(template='seaborn',width=800, height=800)
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
grouped_multiple['World_growth_rate']=grouped_multiple['World_growth_rate'].shift(+1)



fig = go.Figure()
fig.update_layout(template='seaborn',width=800, height=800)
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


mortality = mortality.groupby(['Date', 'Country']).agg({'Deaths': ['sum'],'Recovered': ['sum'],'Confirmed': ['sum']})
mortality.columns = ['Deaths','Recovered','Confirmed']
mortality = mortality.reset_index()
mortality = mortality[mortality.Deaths != 0]
mortality = mortality[mortality.Confirmed != 0]
#prevent division by zero
def ifNull(d):
    temp=1
    if d!=0:
        temp=d
    return temp

mortality['mortality_rate'] = mortality.apply(lambda row: ((row.Deaths+1)/ifNull((row.Confirmed)))*100, axis=1)

#We filter out where mortality rate is above 10% 
d = mortality[mortality.mortality_rate < 10]

#We wannt only to plot countries with more than 50000 confirmed cases, as the situation evovles, more countries will enter this list.
dd = d[d.Confirmed > 50000]

fig = px.line(dd, x="Date", y="mortality_rate", color="Country",
              line_group="Country", hover_name="Country",width=800, height=800)
fig.update_layout(template='seaborn')
fig.show()

##add average



# In[ ]:


cases = data.copy()


cases = cases.groupby(['Date', 'Country']).agg({'Deaths': ['sum'],'Recovered': ['sum'],'Confirmed': ['sum']})
cases.columns = ['Deaths','Recovered','Confirmed']
cases = cases.reset_index()
cases = cases[cases.Deaths != 0]
cases = cases[cases.Country != 'Mainland China']



cases = cases[cases.Confirmed != 0]

#We wannt only to plot countries with more than 50000 confirmed cases, as the situation evovles, more countries will enter this list.

cases = cases[cases.Confirmed > 50000]
cases = cases
#prevent division by zero
def ifNull(d):
    temp=1
    if d!=0:
        temp=d
    return temp

fig = px.line(cases, x="Date", y="Confirmed", color="Country",
              line_group="Country", hover_name="Country", log_y=True,width=800, height=800)
fig.update_layout(template='seaborn')
fig.show()

##add average



# Trend comparison
