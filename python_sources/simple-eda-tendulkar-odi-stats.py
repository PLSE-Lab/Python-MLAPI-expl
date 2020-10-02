#!/usr/bin/env python
# coding: utf-8

# # Univariate analysis

# In[ ]:


import pandas as pd


# In[ ]:


df=pd.read_csv("../input/tendulkar_ODI.csv")


# In[ ]:


df


# #### on Categorical Data

# In[ ]:


import plotly
import plotly.graph_objs as go
from plotly.offline import iplot
plotly.offline.init_notebook_mode(connected=True)


# In[ ]:


trace=go.Histogram(
x=list(df['Dismissal']))
data=[trace]
iplot(data)


# In[ ]:


trace=go.Bar(
x=list(df['Dismissal'].value_counts().index),
y=list(df['Dismissal'].value_counts().values))
data=[trace]
iplot(data)


# In[ ]:


trace=go.Pie(
labels=list(df['Dismissal'].value_counts().index),
values=list(df['Dismissal'].value_counts().values))
data=[trace]
iplot(data)


# ### On Numerical Data

# In[ ]:


print(df.describe(include='all'))


# In[ ]:


trace=go.Box(
y=list(df['4s']),
    boxpoints='all',
        jitter=0.3,
        pointpos=-1.8
)

trace1=go.Box(
y=list(df['6s']),
    boxpoints='all',
        jitter=0.3,
        pointpos=-1.8
)

data=[trace,trace1]
fig = dict(data=data)
iplot(fig)

