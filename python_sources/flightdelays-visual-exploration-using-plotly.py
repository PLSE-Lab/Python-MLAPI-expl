#!/usr/bin/env python
# coding: utf-8

# ### In this kernel, we will visually explore the flight delays dataset, mosty using plotly.
# ### Plots
# - 01 Check the target variable dep_delayed_15min
# - 02 Plot the UniqueCarrier and their frequencies.
# - 03  Plot the UniqueCarrier and the Delay.
# - 04 Plot histogram for Distance.
# - 05 Plot the histogram of Distance with Delay
# - 06 Plot the departure hour and the delay
# - 07 Various Boxplots
# - 08 Check the Distrubution
# - 09 Scatter PLot
# - 10 Parallel Coordinates

# In[ ]:


# Import the libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Viz with Plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly_express as px

# Seaborn and matplotlib
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# In[ ]:


# Read the train file
df_train = pd.read_csv('../input/flight_delays_train.csv')


# In[ ]:


# Check the dimensions
df_train.shape


# In[ ]:


#View first few rows
df_train.head()


# In[ ]:


# Clean month, day of month and day of week
df_train['Month'] = df_train['Month'].str[2:].astype('int')
df_train['DayofMonth'] = df_train['DayofMonth'].str[2:].astype('int')
df_train['DayOfWeek'] = df_train['DayOfWeek'].str[2:].astype('int')

# Check the results
df_train.sample(10)


# ### 01 Check the target variable dep_delayed_15min

# In[ ]:


# Check the target variable
trace = [go.Bar(
            x = df_train['dep_delayed_15min'].value_counts().index.values,
            y = df_train['dep_delayed_15min'].value_counts().values,
            #x = levels,
            #text='Distribution of target variable',
            marker = dict(color='red', opacity=0.6)
    )]


layout = dict(title="Target variable,15 min Delay, distribution", 
              margin=dict(l=100), 
              width=400, 
              height=400)

fig = go.Figure(data=trace, layout=layout)

iplot(fig)


# The target is not so badly skewed, the 15 mins delay(Y) has around 20 k rows.

# ### 02 Plot the UniqueCarrier and their frequencies.

# In[ ]:


# Plot the UniqueCarrier wise frequency of flights
# UniqueCarrier
trace = [go.Bar(
            x = df_train['UniqueCarrier'].value_counts().index.values,
            y = df_train['UniqueCarrier'].value_counts().values,
            marker = dict(color='blue', opacity=0.6)
    )]


layout = dict(title="Carrier wise flight distribution", 
              width=800, 
              height=400,
                 xaxis=dict(title='Unique Carrier',tickmode='linear',tickangle=-45))

fig = go.Figure(data=trace, layout=layout)

iplot(fig)


# There is lot of difference between the head and tail in the frequencies.If we plan to use this variable in our model, potentiialy we can see if we can merger a few of them together.

# ### 03  Plot the UniqueCarrier and the Delay.

# In[ ]:


# By UniqueCarrier and Delay

trace1 = go.Bar(
            x = df_train[df_train['dep_delayed_15min'] == 'Y']['UniqueCarrier'].value_counts().index.values,
            y = df_train[df_train['dep_delayed_15min'] == 'Y']['UniqueCarrier'].value_counts().values,
            name='Yes',
            #marker=dict(color='rgb(49,130,189)')
            marker=dict(color='red',opacity=0.6)
)
trace2 = go.Bar(
            x = df_train[df_train['dep_delayed_15min'] == 'N']['UniqueCarrier'].value_counts().index.values,
            y = df_train[df_train['dep_delayed_15min'] == 'N']['UniqueCarrier'].value_counts().values,
            name='No',
            marker=dict(color='grey',opacity=0.8)
)

data = [trace1, trace2]
    
layout = go.Layout(title="Carrier wise flight distribution by Delay",
    xaxis=dict(title='Unique Carrier',tickangle=-45),
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ### 04 Plot histogram for Distance.

# In[ ]:


# Explore the Distance variable
data = [go.Histogram(x=df_train['Distance'])]
iplot(data)


# There are quite a few outliers.

# ### 05 Plot the histogram of Distance with Delay. 

# In[ ]:


# Distance and Delay
trace1 = go.Histogram(
    x=df_train[df_train['dep_delayed_15min'] == 'Y']['Distance'],
    name='Yes',
    marker=dict(color='red',opacity=0.6)
)
trace2 = go.Histogram(
    x=df_train[df_train['dep_delayed_15min'] == 'N']['Distance'],
    name='No',
    marker=dict(color='blue',opacity=0.2)
)

data = [trace1, trace2]
layout = go.Layout(title="Distance travelled and Delay",
                   barmode='overlay')
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Observe less delays in long distance flights.

# ### Create new features

# In[ ]:


# Create New Features - Departure hour and minute from Departure time
df_train['Dep_hour'] =  df_train['DepTime']//100
df_train['Dep_minute'] =  df_train['DepTime']%100
df_train['Dep_hour'].replace(to_replace=[24,25], value=0, inplace=True)


# ### 06 Plot the departure hour and the delay.

# In[ ]:


df_t = pd.crosstab(df_train.Dep_hour,df_train.dep_delayed_15min)
df_t.head()


# In[ ]:


trace0 = go.Scatter(
    x = df_t.index,
    y = df_t['N'],
    mode = 'lines+markers',
    name = 'No Delay'
)
trace1 = go.Scatter(
    x = df_t.index,
    y = df_t['Y'],
    #mode = 'markers',
    mode = 'lines+markers',
    name = 'Delays'
)
data = [trace0, trace1]
layout = go.Layout(title="Departure Hour and Delay")

fig = go.Figure(data=data, layout=layout)

iplot(fig)


# The delays happen between hours 6 and 23.Lets create a flag to denote this behaviour.

# In[ ]:


# Create a new flag 
df_train['Dep_hour_flag'] = ((df_train['Dep_hour'] >= 6) & (df_train['Dep_hour'] < 23)).astype('int')
df_t = pd.crosstab(df_train.Dep_hour_flag,df_train.dep_delayed_15min)
df_t.head()


# In[ ]:


trace0 = go.Scatter(
    x = df_t.index,
    y = df_t['N'],
    mode = 'lines+markers',
    name = 'No Delay'
)
trace1 = go.Scatter(
    x = df_t.index,
    y = df_t['Y'],
    #mode = 'markers',
    mode = 'lines+markers',
    name = 'Delays'
)
data = [trace0, trace1]
layout = go.Layout(title="Departure Hour and Delay")

fig = go.Figure(data=data, layout=layout)

iplot(fig)


# Potential we can use this in our model.

# ### 07 Boxplots

# In[ ]:


df_train.boxplot(column="Distance",by= "Dep_hour",figsize= (20,10));


# The mean distance travelled per hour is different.The longest flights start first thing in the morning.

# In[ ]:


f, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(x='UniqueCarrier', y = 'Distance',ax = ax, data = df_train, hue = 'dep_delayed_15min' )
ax.set_title('Distance Travelled by Carriers', size = 25)


# WN has most number of flights and seems to fly over shorter distances than AA which has the second most frequency of flights.

# In[ ]:


f, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(x='UniqueCarrier', y = 'DepTime',ax = ax, data = df_train, hue = 'dep_delayed_15min' )
ax.set_title('Departure Time by Carriers', size = 25)


# The mean time of all the flights which are delayed is later than the flights which are not delayed.

# In[ ]:


f, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(x='UniqueCarrier', y = 'Dep_hour',ax = ax, data = df_train, hue = 'dep_delayed_15min' )
ax.set_title('Departure Hour by Carriers', size = 25);


# Same as the above plot but just the hour component.

# ### 08 Check the Distrubution

# In[ ]:


# Define a functoin ot create the distribution plots

def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue = target, aspect = 4, row = row, col = col)
    facet.map(sns.kdeplot, var, shade = True)
    facet.set(xlim = (0, df[var].max()))
    facet.add_legend()
    plt.show()


# In[ ]:


#select numeric columns
num_cols = ['Distance', 'DepTime', 'Dep_hour','Dep_minute']
for numy in num_cols:
    plot_distribution(df_train, numy, 'dep_delayed_15min')


# Again we observe the difference in hour distribution for delayed and ontime flights.

# In[ ]:


# Using plotly express
px.box(df_train, x="Distance", y="UniqueCarrier", orientation="h")


# ### 09 Scatter plot

# In[ ]:


trace0 = go.Scatter(
    x = df_train[df_train['dep_delayed_15min'] == 'Y']['Distance'],
    y = df_train[df_train['dep_delayed_15min'] == 'Y']['DepTime'],
    name = 'Yes',
    mode = 'markers',
    marker = dict(
        size = 30,
        color = rgba(152, 0, 0, .8)',
        line = dict(
            width = 2,
            color = 'rgb(0, 0, 0)'
        )
    )
)

trace1 = go.Scatter(
    x = df_train[df_train['dep_delayed_15min'] == 'N']['Distance'],
    y = df_train[df_train['dep_delayed_15min'] == 'N']['DepTime'],
    name = 'No',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = rgba(255, 182, 193, .6)',
        line = dict(
            width = 2,
        )
    )
)

data = [trace0, trace1]

layout = dict(title = 'Distance and Departure Time',
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False)
             )

fig = dict(data=data, layout=layout)
iplot(fig)


# In[ ]:


df_t = df_train[['dep_delayed_15min','Month','DayOfWeek','Dep_hour']]


dayOfWeek={1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 6:'Saturday', 7:'Sunday'}
df_t['DayOfWeek'] = df_t['DayOfWeek'].map(dayOfWeek)

mon_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
df_t['Month'] = df_t['Month'].map(mon_map)
df_t.head()


# In[ ]:


px.parallel_categories(df_t)


# In[ ]:


px.parallel_categories(df_t, color="Dep_hour")


# In[ ]:


df_t.head()

