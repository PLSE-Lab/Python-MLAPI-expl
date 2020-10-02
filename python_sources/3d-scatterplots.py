#!/usr/bin/env python
# coding: utf-8

# So in this notebook, i am trying my hand on  3d plots by plotly (A great data visualization tool).

# In[ ]:


#Let's load up out data visualization tools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly.graph_objs import *



# In[ ]:


menu = pd.read_csv('../input/menu.csv')
menu.head()


# In[ ]:


#let's check for any null
menu.isnull().sum()


# So the data is perfect we don't need any type of data cleaning

# We gonna plot a 3d scatter plot where **X-axis** will contain the item names  , and the other two axis will have unhealthy nutrients 
# 
#  1. so too much Cholesterol is bad for us, so lets put this on our **Y axis**
#  2. saturated fat is often called bad fat, so let's put this it on our **Z axis**
# 
# 

# In[ ]:



trace1 = go.Scatter3d(
    x=menu['Item'].values,
    y=menu['Cholesterol (% Daily Value)'].values,
    z=menu['Saturated Fat (% Daily Value)'].values,
    mode='markers',
    marker=dict(
        size=12,
        color=menu['Sodium (% Daily Value)'].values,# set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
    
)

data = [trace1]
layout = go.Layout(
    scene=Scene(
       
        yaxis=YAxis(title='Cholesterol (% Daily Value)'),
        zaxis=ZAxis(title='Saturated Fat (% Daily Value)')
        ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter-colorscale')


# In[ ]:





# let's check for some healthy nutrients

# In[ ]:


trace1 = go.Scatter3d(
    x=menu['Item'].values,
    y=menu['Protein'].values,
    z=menu['Calcium (% Daily Value)'].values,
    mode='markers',
    marker=dict(
        size=12,
        color=menu['Sodium (% Daily Value)'].values,# set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
    
)

data = [trace1]
layout = go.Layout(
    scene=Scene(
       
        yaxis=YAxis(title='Protein'),
        zaxis=ZAxis(title='Calcium (% Daily Value)')
        ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter-colorscale')


# In[ ]:




