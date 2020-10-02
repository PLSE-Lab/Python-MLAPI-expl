#!/usr/bin/env python
# coding: utf-8

# # COVID-19: EDA & Time Series Analysis
# Featuring Time Series, Line Charts, Bar Charts, Pie Charts interactive visualizations in Plotly using Johns Hopkins Dataset
# 
# COVID 19 has been in the news and on all of the social media platforms out there. Here's the EDA and Time Series analysis using pie charts, bar graphs, scatter plots and line charts.<br>
# 
# This dataset is managed by the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE) which is supported by ESRI Living Atlas Team and the Johns Hopkins University Applied Physics Lab (JHU APL).

# Ploty has deprecated plotly.plot package. We wil install chart-studio to replace plotly.plot package.

# In[ ]:


get_ipython().system('pip install chart-studio')


# In[ ]:


from IPython.display import Image
import chart_studio.plotly as py

Image("../input/covid-19-image/COVID19.jpg")


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import chart_studio.plotly as py
from plotly.graph_objs import *
from IPython.display import Image
pd.set_option('display.max_rows', None)

import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

# For Density plots
from plotly.tools import FigureFactory as FF

import datetime
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Read the dataset files

# In[ ]:


COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")
COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")
covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")


# ### Observe the data

# Read the contents of each CSV using df.head() to get insights into the fields of the dataframes and understand data represented by each dataframe

# In[ ]:


covid_19_data.head()


# In[ ]:


time_series_covid_19_confirmed.head()


# In[ ]:


time_series_covid_19_deaths.head()


# In[ ]:


time_series_covid_19_recovered.head()


# ### Observe the spread across the countries

# Group by 'covid_19_data' over 'Country/Region' columns to visualize the total number of cases across countries

# In[ ]:


def group_covid(data,country):
    cases = data.groupby(country).size()
    cases = np.log(cases)
    cases = cases.sort_values()
    
    # Visualize the results
    fig=plt.figure(figsize=(35,7))
    plt.yticks(fontsize=8)
    cases.plot(kind='bar',fontsize=12,color='orange')
    plt.xlabel('')
    plt.ylabel('Number of cases',fontsize=10)

group_covid(covid_19_data,'Country/Region')


# # Line Chart with Markers

# Observe the cases across the countries with the help of line charts with markers
# Get the actual values for the group by result

# In[ ]:


cases = covid_19_data.groupby(['Country/Region']).size()
cases = cases.sort_values()

grouped_df = pd.DataFrame(cases)

grouped_df['Count'] = pd.Series(cases).values
grouped_df['Country/Region'] = grouped_df.index
grouped_df['Log Count'] = np.log(grouped_df['Count'])
grouped_df.head()


# ### Visualize the data points using line charts

# Visualize the results by sorting the cases in ascending order and plotting them on line chart to observe the data distribution

# In[ ]:


fig = go.Figure(go.Scatter(
    x = grouped_df['Country/Region'],
    y = grouped_df['Count'],
    text=['Line Chart with lines and markers'],
    name='Country/Region',
    mode='lines+markers',
    marker_color='#56B870'
))

fig.update_layout(
    height=800,
    title_text='COVID-19 cases across nations using line chart',
    showlegend=True
)

fig.show()


# Visualize the cases aross countries in by normalizing the values using log scale

# In[ ]:


fig = go.Figure(go.Scatter(
    x = grouped_df['Country/Region'],
    y = grouped_df['Log Count'],
    text=['Line Chart with lines and markers'],
    name='Country/Region',
    mode='lines+markers',
    marker_color='#56B870'
))

fig.update_layout(
    height=800,
    title_text='COVID-19 cases across nations using line chart (Log scale)',
    showlegend=True
)

fig.show()


# # Bar chart

# Since we are dealing with enormous COVID-19 data, we need to scale the data so that it fits the visualization
# Colors of the bar charts are based on the logarithmic value of the cases for each country

# In[ ]:


fig = go.Figure(go.Bar(
    x = grouped_df['Country/Region'],
    y = grouped_df['Log Count'],
    text=['Bar Chart'],
    name='Countries',
    marker_color=grouped_df['Count']
))

fig.update_layout(
    height=800,
    title_text='COVID-19 cases across nations using bar chart (Log scale)',
    showlegend=True
)

fig.show()


# We perform logarithmic scaling over the 'Cases' column and then sort the values so that 
# we can have the bars arranged alphabetically

# In[ ]:


fig = go.Figure(go.Bar(
    x = grouped_df['Country/Region'],
    y = grouped_df['Log Count'],
    text=['Bar Chart'],
    name='Countries',
    marker_color=grouped_df['Log Count']
))

fig.update_layout(
    height=800,
    title_text='COVID-19 cases across nations using bar chart (Log scale)',
    showlegend=True
)

fig.show()


# # Perform Time Series Analysis

# Read the contents of grouped_df to comprehend the data and the fields in the dataframe

# In[ ]:


grouped_df.head()


# ### Observe the spread of Covid across different nations across times

# Group by cases across countries to visualize the data over 'ObservationDate'

# In[ ]:


covid_countries = covid_19_data.groupby('Country/Region')['ObservationDate'].value_counts().reset_index(name='t')
covid_countries['Count'] = covid_19_data.groupby('Country/Region')['ObservationDate'].transform('size')

covid_countries.head()


# In[ ]:


# Create traces
fig = go.Figure()

fig.add_trace(go.Scatter(
    x = covid_countries['ObservationDate'],
    y = covid_countries['Country/Region'],
    text=['Line Chart with lines and markers'],
    name='Countries',
    mode='markers',
    
))

fig.update_layout(
    height=800,
    title_text='COVID-19 case occurences across nations',
    showlegend=True
)

fig.show()


# # Observe the cases over time

# The data points represent the fact that a case was recorded for a given nation on a particular date

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(x = covid_countries['ObservationDate'],
                y = covid_countries['Country/Region'],
                name='Country/Region',
                marker_color=covid_countries['Count']
                ))

fig.update_layout(
    height=800,
    title_text='COVID-19 cases visualized over time across nations',
    
    showlegend=True
)

fig.show()


# # Visualize the composition of the cases

# ### Get 'Confirmed' cases for each country

# In[ ]:



confirmed_df = covid_19_data.groupby('Confirmed')['Country/Region'].value_counts().reset_index(name='Conf')
confirmed_df['Conf'] = covid_19_data.groupby('Confirmed')['Country/Region'].transform('size')

confirmed_df.head()


# ### Visualize Confirmed cases over Pie Chart

# In[ ]:


fig = px.pie(confirmed_df, values='Conf', names='Country/Region')
fig.show()


# ### Get 'Deaths' cases for each country

# In[ ]:


deaths_df = covid_19_data.groupby('Deaths')['Country/Region'].value_counts().reset_index(name='Death')
deaths_df['Death'] = covid_19_data.groupby('Deaths')['Country/Region'].transform('size')

deaths_df.head()


# ### Visualize 'Deaths' for each country

# In[ ]:


fig = px.pie(deaths_df, values='Death', names='Country/Region')
fig.show()


# ### Get 'Recovered' cases for each country

# In[ ]:


recover_df = covid_19_data.groupby('Recovered')['Country/Region'].value_counts().reset_index(name='Recover')
recover_df['Recover'] = covid_19_data.groupby('Recovered')['Country/Region'].transform('size')

recover_df.head()


# ### Visualize 'Recovered' cases over Pie Chart

# In[ ]:


fig = px.pie(recover_df, values='Recover', names='Country/Region')
fig.show()


# # Density Graphs

# In[ ]:


sns.color_palette("cubehelix", 8)
sns.set_style("whitegrid", {'axes.grid' : False})

sns.color_palette("cubehelix", 8)
sns.distplot(confirmed_df['Conf'],bins=100,hist=False,   label="Confirmed Cases");
# sns.distplot(deaths_df['Death'],bins=100,hist=False,   label="Deaths");
# sns.distplot(recover_df['Recover'],bins=100,hist=False,   label="Recovered Cases");


plt.legend();


# In[ ]:




