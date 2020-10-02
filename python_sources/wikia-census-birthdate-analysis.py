#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pylab
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)


# # Load data

# In[ ]:


df = pd.read_csv("../input/20181019-wikia_stats_users_birthdate.csv")
df['datetime.birthDate'] = pd.to_datetime(df['datetime.birthDate'], infer_datetime_format=True, errors='coerce') 
df.set_index(df['datetime.birthDate'], inplace=True)
df.head()


# # Number of wikis over the years

# In[ ]:


byYear = df.resample('y').count()['id']
byYear


# Data shows an anomaly because Wikia was created in 2004. A deepen analysis of some of the anomaly wikis shows that some fake birthdates were included by a maintenance script (see [the earliest edit of this wiki, as an example](http://blasterman.wikia.com/wiki/Blasterman_Wiki?dir=prev&action=history)).
# 
# Additionally, 2018 wikis are very young (no more than two months) so they will be also removed.
# 

# In[ ]:


dfClean = df['2004':'2017'].copy()
byYear = dfClean.resample('y').count()['id']

# Active Wikis: at least one active user in the last 30 days
activeByYear = dfClean[(dfClean['stats.activeUsers']>=1)&(dfClean['users_1']>0)].resample('y').count()['id']
activeByYear


# In[ ]:


traceTotal = go.Bar(x=byYear.index.year, y=byYear.values, name="Total wikis")
traceActive = go.Bar(x=activeByYear.index.year, y=activeByYear.values, name="Active wikis")
layout = go.Layout(
    legend=dict(x=0.1, y=0.85),
    xaxis=dict(
        tickangle=30
    )
)
iplot(go.Figure(data=[traceTotal, traceActive], layout=layout), filename='byYear')


# ## Wiki age
# 
# Instead of showing the number of wikis over the years, we will focus on the age of the active wikis in order to visualize the population pyramid.

# In[ ]:


def computeAge(birthDate):
    timeSinceBirth = pd.Timestamp(2018, 2, 20)-birthDate
    return int(timeSinceBirth.days/365)
dfClean['age'] = dfClean['datetime.birthDate'].apply(computeAge)


# In[ ]:


activeWikis = dfClean[(dfClean['stats.activeUsers']>=1)&(dfClean['users_1']>0)]
inactiveWikis = dfClean[(dfClean['stats.activeUsers']<1)|(dfClean['users_1']==0)]


# In[ ]:


activeByAge = activeWikis.groupby(by=['age']).url.count()
inactiveByAge = inactiveWikis.groupby(by=['age']).url.count()


# In[ ]:


trace0 = go.Scatter(
    x=activeByAge.index.values,
    y=activeByAge.values,
    mode='lines',
    name="Active wikis",
    line=dict(width=0.5),
    fill='tonexty'
)

trace1 = go.Scatter(
    x=inactiveByAge.index.values,
    y=inactiveByAge.values,
    mode='lines',
    name="Inactive wikis",
    line=dict(width=0.5),
    fill='tonexty'
)

layout = go.Layout(
    yaxis=dict(title='Number of active wikis'),
    xaxis=dict(
        domain=[0,0.5],
        tickmode='array',
        tickvals=list(range(0,20)),
        title="Age (in years)"
    ),
    legend=dict(
        x=0.4
    )
);

fig = go.Figure(data=[trace0], layout=layout)
iplot(fig, filename='stacked-area-plot')


# In[ ]:




