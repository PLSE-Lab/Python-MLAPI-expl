#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Part of the Kaggle Professional Series event by Rachael Tatman
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# In[ ]:


red_light_vio = pd.read_csv('../input/red-light-camera-violations.csv')


# In[ ]:


# Violation date is string so we need to change it to datetime to make use of python's date time indexing features
red_light_vio['VIOLATION DATE'] = pd.to_datetime(red_light_vio['VIOLATION DATE'])
red_light_vio.sort_values(by='VIOLATION DATE',inplace=True)
red_light_vio.set_index('VIOLATION DATE',inplace=True)


# In[ ]:


curr_yr_tot = red_light_vio.loc['{}'.format(pd.datetime.now().year)].groupby(['INTERSECTION']).VIOLATIONS.sum()


# In[ ]:


# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
top_ten_curr_year  = curr_yr_tot.sort_values(ascending=False)[0:20]
x=top_ten_curr_year.index.tolist()
y=top_ten_curr_year.values
data = [go.Bar(x=x,y=y,hoverinfo=('x','y'))]

# specify the layout of our figure
layout = dict(title = "Top 20 redlight violation locations in the current year",
              xaxis= dict(title= 'Intersection',ticklen= 3,zeroline= True,automargin=True))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


tot_violations_per_year = red_light_vio.groupby(lambda x: x.year).VIOLATIONS.sum()
# plt.title('Total Red Light Violation per Year')
# plt.show()

x= tot_violations_per_year.index.tolist()
y= tot_violations_per_year.values
data = [go.Scatter(x=x,y=y,
                  mode='lines+markers',
                  hoverinfo=('x','y'))]

# specify the layout of our figure
layout = dict(title = "Top 20 redlight violation locations in the current year",
             xaxis= dict(title= 'Year',tickvals = x,ticklen= 3,zeroline= True,automargin=True))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)


# 
