#!/usr/bin/env python
# coding: utf-8

# <img src="https://www.statnews.com/wp-content/uploads/2020/02/Coronavirus-CDC.jpg">
# 
# 
# 
# <font face = "Verdana" size ="5"> WHO NEEDS AN INTRODUCTION OF THE MONSTER COVID19 HERE?

# #### Importing packages

# In[ ]:


import pandas as pd
import numpy as np

import plotly.express as px
from plotly.graph_objs import *
import plotly.graph_objects as go


# #### Importing data

# In[ ]:


df= pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
df.head()


# In[ ]:


df.info()


# ## Data Preprocessing

# In[ ]:


# Changing datatypes

df['ObservationDate']=pd.to_datetime(df['ObservationDate'])
df['Confirmed']=df['Confirmed'].astype(int)
df['Deaths']=df['Deaths'].astype(int)
df['Recovered']=df['Recovered'].astype(int)

# Fetching latest stats 

current_date= df['ObservationDate'].max()

current= df.loc[df['ObservationDate']==current_date]
current.reset_index(drop=True, inplace=True)


# ## Grouping data by countries

# In[ ]:


group=current.groupby(['Country/Region'])['Confirmed','Deaths','Recovered' ].sum().reset_index()
group.head()


# # Lets dive deep into data!! 
# 
# 
# ## Lets have an overview of how much this virus has affected the whole world!!!

# In[ ]:


fig = px.choropleth(group, locations="Country/Region",locationmode='country names',
                    color="Confirmed", # lifeExp is a column of gapminder
                    hover_name='Country/Region', # column to add to hover information
                    hover_data=['Confirmed'],
                    labels={'Confirmed':'Confirmed Cases', 
                           'Country/Region':'Country'},
                    color_continuous_scale=px.colors.sequential.amp)

fig.update_geos(projection_type="orthographic")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ## Total Affected, Recovered and Deaths in the world!!

# In[ ]:


total_cases= sum(current.Confirmed)
total_Deaths= sum(current.Deaths)
total_Recovered= sum(current.Recovered)


layout = Layout(
    template='plotly_white'
)

fig = go.Figure(layout=layout)
fig.add_trace(go.Bar(
    y=['COVID19'],
    x=[total_cases],
    name='Confirmed',
    orientation='h',
    marker=dict(
        color='rgba(199, 8, 46, 0.8)',
        line=dict(color='rgba(199, 8, 46, 1.0)', width=3)
    )
))
fig.add_trace(go.Bar(
    y=['COVID19'],
    x=[total_Recovered],
    name='Recovered',
    orientation='h',
    marker=dict(
        color='rgba(242, 78, 110, 0.9)',
        line=dict(color='rgba(242, 78, 110, 1.0)', width=3)
    )
))

fig.add_trace(go.Bar(
    y=['COVID19'],
    x=[total_Deaths],
    name='Deaths',
    orientation='h',
    marker=dict(
        color='rgba(58, 71, 80, 0.8)',
        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
    )
))

fig.update_layout(barmode='stack', height=300, width=980,)
fig.show()


# ## Lets transform data to have a view at trends!!!

# In[ ]:


trend=df.groupby(['ObservationDate'])['Confirmed','Deaths','Recovered' ].sum().reset_index()


new_confirmed=list(np.diff(trend['Confirmed'].to_list()))
new_deaths=list(np.diff(trend['Deaths'].to_list()))
new_rec=list(np.diff(trend['Recovered'].to_list()))


trend['New_confirmed']=trend.loc[0, 'Confirmed']
trend['New_Deaths']=trend.loc[0, 'Deaths']
trend['New_Recovered']=trend.loc[0, 'Recovered']
trend.loc[1:,'New_confirmed']=new_confirmed
trend.loc[1:,'New_Deaths']=new_deaths
trend.loc[1:,'New_Recovered']=new_rec
trend.head()


# ## Cumulative stats of affected, recovered and deaths till now!!!

# In[ ]:


layout = Layout(
    template='plotly_white', height=500, width=900,
)


fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(x=trend['ObservationDate'], 
                        y=trend['Confirmed'],
                        line = dict(color='firebrick', width=3),
                        name='Confirmed'))

fig.add_trace(go.Scatter(x=trend['ObservationDate'], 
                        y=trend['Recovered'],
                        line = dict(color='grey', width=3),
                        name='Recovered'))

fig.add_trace(go.Scatter(x=trend['ObservationDate'], 
                        y=trend['Deaths'],
                        line = dict(color='royalblue', width=3),
                        name='Deaths'))

fig.update_layout(title='Cumulative Trend of COVID19 in the World',
                   yaxis_title='Cases')


fig.show()


# ## Daily trend of Affected, Recovered and Deaths till now!!

# In[ ]:


layout = Layout(
    template='plotly_white', height=500, width=900,
)


fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(x=trend['ObservationDate'], 
                        y=trend['New_confirmed'],
                        line = dict(color='firebrick', width=3),
                        name='Confirmed'))

fig.add_trace(go.Scatter(x=trend['ObservationDate'], 
                        y=trend['New_Recovered'],
                        line = dict(color='grey', width=3),
                        name='Recovered'))

fig.add_trace(go.Scatter(x=trend['ObservationDate'], 
                        y=trend['New_Deaths'],
                        line = dict(color='royalblue', width=3),
                        name='Deaths'))

fig.update_layout(title='Daily Trend of COVID19 in the World',
                   yaxis_title='Cases')


fig.show()


# ## Lets find out highly affected countries!!

# In[ ]:


group=current.groupby(['Country/Region'])['Confirmed','Deaths','Recovered' ].sum().reset_index()

group.sort_values(by='Confirmed', ascending=False, inplace=True)
group.reset_index(drop=True, inplace=True)

total_cases= sum(group['Confirmed'])
group['Confirmed_Ratio'] = group['Confirmed'].apply(lambda x: round((x/total_cases),2))

group= group.loc[:20]

group


# In[ ]:


names=group['Country/Region'].to_list()
Confirmed=group['Confirmed'].to_list()
Ratio_Confirmed=group['Confirmed_Ratio'].to_list()
Recovered=group['Recovered'].to_list()
deaths=group['Deaths'].to_list()
base=[ i*-1 for i in deaths]


# ## Countries having huge no of affected cases!!

# In[ ]:


layout = Layout(template='plotly_white',
                height=450,
                width=980)

fig1 = go.Figure(layout=layout)

fig1.add_trace(go.Bar(x=names, y=Confirmed,
                base=0,
                marker_color='crimson',
                name='Confirmed',
                         marker=dict(
        color='rgba(168, 7, 39, 0.6)',
        line=dict(color='rgba(168, 7, 39, 1.0)', width=3)
    )))
fig1.update_layout(title='Confirmed Cases - Top 20 Countries')

fig1.show()


fig2 = go.Figure(layout=layout)

fig2.add_trace(go.Bar(x=names, y=Ratio_Confirmed,
                base=0,
                marker_color='crimson',
                name='Ratio_Confirmed',
                         marker=dict(
        color='rgba(168, 7, 39, 0.6)',
        line=dict(color='rgba(168, 7, 39, 1.0)', width=3)
    )
                ))
fig2.update_layout(yaxis_tickformat = '%',
                  title='Percent in Total Confirmed Cases of Top 20 Countries')

fig2.show()


# ## Lets view at how many of them have recovered!!

# In[ ]:


fig3 = go.Figure(layout=layout)


fig3.add_trace(go.Bar(x=names, y=Recovered,
                base=0,
                marker_color='lightslategrey',
                name='Recovered',
                         marker=dict(
        color='rgba(58, 71, 80, 0.6)',
        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
    )
                ))
fig3.add_trace(go.Bar(x=names, y=deaths,
                base=base,
                marker_color='crimson',
                name='Deaths',
                        marker=dict(
        color='rgba(168, 7, 39, 0.6)',
        line=dict(color='rgba(168, 7, 39, 1.0)', width=3)
    )))
fig3.update_layout(title='Recovered and Deaths - Top 20 Countries')

fig3.show()


# #### Updating...
