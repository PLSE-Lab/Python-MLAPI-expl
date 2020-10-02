#!/usr/bin/env python
# coding: utf-8

# # Video Games Analysis and Visualization

# In this notebook, first I analyze the data and then visualize it.

# **Importing Libraries**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt
import plotly.express as px


# In[ ]:


data = pd.read_csv('../input/videogamesales/vgsales.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# Here I have done some preprocessing on our dataset.

# In[ ]:


data.dropna(how="any",inplace = True)
data.info()


# In[ ]:


data.Year = data.Year.astype(int)
data.head()


# Here preprocessing is finished.

# Let's start making some differnt different visualizations on the given dataset.

# **Set size and color with column names**

# Note that color and size data are added to hover information. You can add other columns to hover data with the hover_data argument of px.scatter.

# Also you can learn this type of plotting by this link : - https://plotly.com/python/line-and-scatter/#set-size-and-color-with-column-names

# In[ ]:


fig = px.scatter(data, x="Year", y="NA_Sales", 
                 color="NA_Sales",
                 size='NA_Sales', 
                 hover_data=['Rank','Name', 'Platform', 'Genre', 'Publisher'], 
                 title = "Sales in North America")
fig.show()


# **Style Scatter Plots**

# You can learn this type of plotting by this link : - https://plotly.com/python/line-and-scatter/#style-scatter-plots

# In[ ]:


import plotly.graph_objects as go
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=data['Year'], y=np.sin(data['EU_Sales']),
    name='sin',
    mode='markers',
    marker_color='rgba(152, 0, 0, .8)'
))
fig1.update_traces(mode='markers', marker_line_width=2, marker_size=10)
fig1.update_layout(title='Sales in Europe',
                  yaxis_zeroline=False, xaxis_zeroline=False)


fig1.show()


# **Data Labels on Hover**

# Here is the link of understanding this type of plotting : -https://plotly.com/python/line-and-scatter/#data-labels-on-hover

# In[ ]:


fig2 = go.Figure(data=go.Scatter(x=data['Year'],
                                y=data['JP_Sales'],
                                mode='markers',
                                marker_color=data['Rank'],
                                text=data['Name'])) # hover text goes here

fig2.update_layout(title='Sales in Japan')
fig2.show()


# **Scatter with a Color Dimension**

# Here is the link of this type of plotting : - https://plotly.com/python/line-and-scatter/#scatter-with-a-color-dimension

# In[ ]:


fig3 = go.Figure(data=go.Scatter(
    y = data['Other_Sales'],
    mode='markers',
    marker=dict(
        size=16,
        color=data['Rank'], #set color equal to a variable
        colorscale='Viridis', # one of plotly colorscales
        showscale=True
    )
))
fig3.update_layout(title='Sales in Other countries')
fig3.show()


# **Bubble Scatter Plots**

# Here is the link of understanding : - https://plotly.com/python/line-and-scatter/#bubble-scatter-plots

# In[ ]:


fig4 = go.Figure(data=go.Scatter(
    x=data['Year'],
    y=data['Global_Sales'],
    mode='markers',
    marker=dict(size=[40, 60, 80, 100],
                color=[0, 1, 2, 3])
))
fig4.update_layout(title="Sales in GLobal")
fig4.show()


# So, **which plot do you like?** Please give me response by comments so that I can do more visualizations 

# Till then  **Enjoy Machine Learning**

# In[ ]:




