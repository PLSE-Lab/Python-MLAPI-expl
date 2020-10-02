#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import seaborn as sns
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
get_ipython().system('ls ../input/')


# In[ ]:


df = pd.read_csv(r'../input/311-service-requests-rodent-baiting.csv', parse_dates = ['Creation Date','Completion Date'])


# In[ ]:


df.head()


# In[ ]:


df = df[df.creationdate.dt.year==2018]


# In[ ]:


df.columns = [col.lower().replace(' ','') for col in df.columns]
df.columns


# In[ ]:


premwithrats = df.groupby('ward')['numberofpremiseswithrats'].sum().sort_values(ascending = False).reset_index()


# In[ ]:


premwithrats.head()


# In[ ]:


data = [go.Bar(x=premwithrats['ward'], y=premwithrats['numberofpremiseswithrats'])]
layout = go.Layout(
    title='Number of Premises with Rats per Ward 2018',
    xaxis=dict(
        title='Ward',
        tickmode='linear',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ))
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


df['completiondate']=pd.to_datetime(df['completiondate'])


# In[ ]:


TimeGroupDF = df.groupby(df.creationdate.dt.month)['numberofpremisesbaited'].sum()


# In[ ]:


TimeGroupDF.plot()


# In[ ]:


pltTS = pd.DataFrame(df.groupby(df.creationdate)['numberofpremisesbaited'].sum())


# In[ ]:


pltTS = pltTS.reset_index()


# In[ ]:


pltTS.dtypes


# In[ ]:


pltTS = pltTS[pltTS.numberofpremisesbaited != pltTS['numberofpremisesbaited'].max()]


# In[ ]:


#pltTS['CreationDate'] = pltTS['CreationDate'].apply(lambda x: str(x))


# In[ ]:


sns.distplot(pltTS['numberofpremisesbaited'], hist = True)


# In[ ]:


# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=pltTS.creationdate, y=pltTS.numberofpremisesbaited)]


# In[ ]:


fig = data#dict(data = data, layout = layout)
iplot(fig)

