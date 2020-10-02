#!/usr/bin/env python
# coding: utf-8

# In the kernel i will analyze the world commodity trade statistics. 
# 
# -	First I will clean the data 
# -	Find the top trader 
# -	Find the top exporters 
# -	Analyzing export and import 
# 
# Feel free to give me some critics! 
# 

# In[ ]:


#importing lib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import os
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
import plotly.graph_objs as go
init_notebook_mode()


# Reading the csv file

# In[ ]:


k = pd.read_csv('../input/commodity_trade_statistics_data.csv')


# Branching off to just the year 2016

# In[ ]:


world = k[k['year']==2016]


# Checking the head to see the columns and values we got. 

# In[ ]:


world.head()


# This data covers 122 different countries. 
# Which proides of with a lot of data. 

# In[ ]:


world['country_or_area'].nunique()


# We can see that there is some missing data.

# In[ ]:


sns.heatmap(world.isnull())


# In[ ]:


trade=world.dropna()
sns.heatmap(trade.isnull())


# In[ ]:


trade.info()


# In[ ]:


top_quantity = trade.drop(columns=['comm_code','commodity']).groupby(by='country_or_area').agg({'quantity':sum})


# In[ ]:


#summing


# In[ ]:


x = top_quantity['quantity'].nlargest(10)


# In[ ]:


let = [Bar(
            y=x,
            x=x.keys(),
            marker = dict(
            color = 'rgba(25, 82, 1, .9)'
            ),
            name = "Contractor's amount earned per project"
    )]
layout1 = go.Layout(
    title="Top 10 Quantity",
    xaxis=dict(
        title='Country',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
               )
    ),
    yaxis=dict(
        title='Total Amount of Quantity',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
myFigure2 = go.Figure(data = let, layout = layout1)
iplot(myFigure2)


# In[ ]:


# Top Exporters 


# In[ ]:


ex = trade[trade['flow']=='Export']


# In[ ]:


top_Export = ex.drop(columns=['comm_code','commodity']).groupby(by='country_or_area').agg({'quantity':sum})


# In[ ]:


g = top_Export['quantity'].nlargest(10)


# In[ ]:


next = [Bar(
            y=g,
            x=g.keys(),
            marker = dict(
            color = 'rgba(25, 82, 1, .9)'
            ),
            name = "Contractor's amount earned per project"
    )]
layout1 = go.Layout(
    title="Top 10 Exporters",
    xaxis=dict(
        title='Country',
        titlefont=dict(
            family='Courier New, monospace',
            size=30,
            color='#7f7f7f'
               )
    ),
    yaxis=dict(
        title='Total Amount of Quantity',
        titlefont=dict(
            family='Courier New, monospace',
            size=22,
            color='#7f7f7f'
        )
    )
)
myFigure2 = go.Figure(data = next, layout = layout1)
iplot(myFigure2)


# ## Pie chart for the values of Export and Import

# In[ ]:


cat = k.groupby('flow')['year'].count()


# In[ ]:


fig = { 
    "data":[{
        "values":cat,
        "labels":cat.keys(),
        "domain": {"x": [0, 1]},
        "name": "Type",
        "hoverinfo":"label+percent+name",
        "hole": .4,
        "type": "pie",
        "textinfo": "value"
    }],
    "layout":{
        "title":"Ratio of Import and Export",
        "annotations": [
            {
                "font": {
                    "size": 13
                },
                "showarrow": False,
                "text": "DISTRIBUTION",
                "x": 0.7,
                "y": 0.5
            }]
    }
}

trace = go.Pie(labels = cat.keys(), 
               values=cat,textinfo='value', 
               hoverinfo='label+percent', 
               textfont=dict(size = 15))
iplot(fig)

