#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Netflix, Inc.is an American media-services provider and production company headquartered in Los Gatos, California, founded in 1997 by Reed Hastings and Marc Randolph in Scotts Valley, California. The company's primary business is its subscription-based streaming service which offers online streaming of a library of films and television programs, including those produced in-house. As of April 2019, Netflix had over 148 million paid subscriptions worldwide, including 60 million in the United States, and over 154 million subscriptions total including free trials. It is available worldwide except in mainland China (due to local restrictions), Syria, North Korea, and Crimea (due to US sanctions). The company also has offices in the Netherlands, Brazil, India, Japan, and South Korea. Netflix is a member of the Motion Picture Association (MPA).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
py.init_notebook_mode(connected = True)


# In[ ]:


data=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles_nov_2019.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# # To know the Non null values

# In[ ]:


data.dtypes


# # To know the Data Types

# In[ ]:


data.rating.value_counts()


# Since Rating is our Target Variable, We have to know the count of each Rating.

# # Percentage of TV Shows to Movies

# In[ ]:


def pie_plot(cnt_srs, colors, title):
    labels=cnt_srs.index
    values=cnt_srs.values
    trace = go.Pie(labels=labels, 
                   values=values, 
                   title=title, 
                   hoverinfo='percent+value', 
                   textinfo='percent',
                   textposition='inside',
                   hole=0.7,
                   showlegend=True,
                   marker=dict(colors=colors,
                               line=dict(color='#000000',
                                         width=2),
                              )
                  )
    return trace

py.iplot([pie_plot(data['type'].value_counts(), ['green', 'red'], 'Type')])


# Based on the Type, Movies are released more than TV Shows in the given dataset.

# # Rating Graph 

# In[ ]:


def pie_plot(cnt_srs, title):
    labels=cnt_srs.index
    values=cnt_srs.values
    trace = go.Pie(labels=labels, 
                   values=values, 
                   title=title, 
                   hoverinfo='percent+value', 
                   textinfo='percent',
                   showlegend=True,
                   marker=dict(colors=plt.cm.viridis_r(np.linspace(0, 1, 14)),
                               line=dict(color='#000000',
                                         width=1.5),
                              )
                  )
    return trace

py.iplot([pie_plot(data['rating'].value_counts(), 'Type')])


# Based on the graph above, TV-MA type of Rating is watched more followed by TV-14 type

# # Bar Graph based on Rating

# In[ ]:


temp_data = data['rating'].value_counts().reset_index()
graph1 = go.Bar(
                x = temp_data['index'],
                y = temp_data['rating'],
                marker = dict(color = 'rgb(204,0,0)',
                              line=dict(color='rgb(0,0,0)',width=2.0)))
layout = go.Layout(template= "plotly_white",title = 'GRAPH BASED ON RATING' , 
                   xaxis = dict(title = 'Rating'), yaxis = dict(title = 'Count'))
fig = go.Figure(data = [graph1], layout = layout)
fig.show()


# This graph confirm the previous data, TV-MA rated shows are available more in Netflix

# In[ ]:


data.rating.describe()


# In[ ]:




