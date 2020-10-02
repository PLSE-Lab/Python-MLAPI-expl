#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from math import pi
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt,mpld3
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import iplot
from plotly import tools
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
from bokeh.io import output_file, output_notebook
from bokeh.plotting import figure, show, reset_output
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel
import bokeh.palettes
from bokeh.transform import cumsum
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv',parse_dates=['Date'])
df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


df.rename(columns={'Lat':'Latitude','Long':'Longitude','ConfirmedCases':'Confirmed'},inplace=True)
df.head()


# In[ ]:


df.fillna(0,inplace=True)
df.head()


# In[ ]:


print(f"Latest Entry: {df['Date'].max()}")


# In[ ]:


print(f"Average confirmed cases: {round(df['Confirmed'].mean())}")
print(f"Average deaths: {round(df['Deaths'].mean())}")


# In[ ]:


latest=df['Date'].max()
df['Latest_Date']=latest
df.head()


# In[ ]:


plot_map=df.groupby(['Latest_Date','Country/Region']).sum().reset_index()
names=plot_map['Country/Region']
values=plot_map['Deaths']
fig = px.choropleth(plot_map, locations=names,
                    locationmode='country names',
                    color=values)
fig.update_layout(title="Death Progression")
fig.show()


# In[ ]:


spread_in_china=df[df['Country/Region']=='China']
spread_in_china


# In[ ]:


fig, ax = plt.subplots(figsize=(15,15),facecolor='darkgrey')
scatter=ax.scatter(spread_in_china['Confirmed'],spread_in_china['Deaths'],s=120,alpha=0.6,marker='*')
ax.grid(color='darkgray', linestyle='solid')
ax.set_title("Increase in deaths with respect to confirmed", size=20)
plt.xlabel("Confirmed",fontsize=18)
plt.ylabel("Dead",fontsize=18)
plt.show()


# In[ ]:


spread_in_italy=df[df['Country/Region']=='Italy']
spread_in_italy


# In[ ]:


output_notebook()
fig = figure(x_axis_type='datetime',
             plot_height=500, plot_width=700,background_fill_color='grey',
             title='Comparison in spread of virus in Italy and China',
             x_axis_label='Date', y_axis_label='Confirmed',
             toolbar_location='right')
fig.step('Date', 'Confirmed', 
         color='white', legend_label='China', 
         source=spread_in_china)
fig.step('Date', 'Confirmed', 
         color='red', legend_label='Italy', 
         source=spread_in_italy)
fig.legend.location = 'top_left'
show(fig)


# In[ ]:


countries = df[df['Country/Region'].isin(['China', 'Italy', 'US','Spain','India','Pakistan','United Kingdom','Germany','France'])].reset_index()
countries = countries.groupby('Country/Region')['Country/Region','Date', 'Confirmed', 'Deaths'].sum().reset_index()
countries


# In[ ]:


fig = go.Figure(data=[
      go.Bar(name='Confirmed', x=countries['Country/Region'], y=countries['Confirmed'],marker_color='blue'),
      go.Bar(name='Deaths', x=countries['Country/Region'], y=countries['Deaths'],marker_color='yellow')
      ],layout=go.Layout(
      title=go.layout.Title(text="Confirmed and dead in major countries")
      ))
template='plotly_dark'
fig.update_layout(template=template)
fig.show()


# In[ ]:


confirmed_cases_total=df.groupby('Date')['Country/Region','Date','Confirmed','Deaths'].sum().reset_index()
confirmed_cases_total


# In[ ]:


fig=go.Figure(data=[
    go.Scatter(x=confirmed_cases_total['Date'],
    y=confirmed_cases_total['Confirmed'],marker_color='pink')]
)
fig.update_layout(title='Confirmed cases with time',font_size=16,yaxis_type='log',template='plotly_dark')
fig.show()


# In[ ]:




