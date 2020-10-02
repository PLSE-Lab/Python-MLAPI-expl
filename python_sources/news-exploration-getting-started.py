#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# based on the work of
# https://www.kaggle.com/pestipeti/simple-eda-two-sigma


# In[ ]:


from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()


# In[ ]:


import numpy as np
import pandas as pd

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


# In[ ]:


mt_df, nt_df = env.get_training_data()


# In[ ]:


print("{:,} news samples".format(nt_df.shape[0]))


# In[ ]:


nt_df.dtypes


# In[ ]:


nt_df.isna().sum()


# In[ ]:


nt_df.nunique()


# In[ ]:


urgency = nt_df.groupby(nt_df['urgency'])['urgency'].count().sort_values(ascending=False)
# Create a trace
trace1 = go.Pie(
    labels = urgency.index,
    values = urgency.values
)

layout = dict(title = "Urgency (1:alert, 3:article)")
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')


# In[ ]:


articlesOnly = nt_df[nt_df['urgency'] == 3]
articleSources = articlesOnly.groupby(articlesOnly['provider'])['provider'].count()
topArticleSources = articleSources.sort_values(ascending=False)[0:10]

# Create a trace
trace1 = go.Pie(
    labels = topArticleSources.index,
    values = topArticleSources.values
)

layout = dict(title = "Top article sources")
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')


# In[ ]:


alertsOnly = nt_df[nt_df['urgency'] == 1]
alertsSources = alertsOnly.groupby(alertsOnly['provider'])['provider'].count()
alertsSources.sort_values(ascending=False)[0:10]


# In[ ]:




