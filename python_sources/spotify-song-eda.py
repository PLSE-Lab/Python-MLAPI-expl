#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
top50 = pd.read_csv("../input/top50spotify2019/top50.csv", encoding = "ISO-8859-1")


# In[ ]:


top50.head()


# In[ ]:


distilledtop50 = top50.drop(['Artist.Name', 'Genre'], axis=1)


# In[ ]:


dt50 = distilledtop50


# In[ ]:


sns.heatmap(distilledtop50.head(20).corr(), cmap='mako')


# In[ ]:


sns.heatmap(distilledtop50.head(20).corr(), cmap='mako', annot=True)


# In[ ]:


sns.heatmap(top50.corr())


# In[ ]:


sns.distplot(dt50["Beats.Per.Minute"], color='g')


# In[ ]:


sns.distplot(dt50["Energy"], color='b')


# In[ ]:


sns.clustermap(dt50.corr(), cmap='mako')


# In[ ]:


sns.clustermap(dt50.corr(), cmap='YlGnBu')


# In[ ]:


sns.clustermap(dt50.corr(), cmap='viridis',annot=True)


# In[ ]:


dt50.head()


# 

# In[ ]:


import plotly.figure_factory as ff

import numpy as np
dt50 = dt50.drop(['Track.Name'], axis=1)
fig = ff.create_dendrogram(dt50, color_threshold=1.5)
fig.update_layout(width=800, height=500)
fig.show()


# In[ ]:


import plotly.graph_objects as go

import numpy as np


fig = go.Figure(data=[go.Histogram(y=top50['Danceability'])])
fig.show()


# In[ ]:


import plotly.graph_objects as go

import numpy as np


fig = go.Figure()
fig.add_trace(go.Histogram(x=top50.Popularity))
fig.add_trace(go.Histogram(x=top50['Valence.']))

fig.add_trace(go.Histogram(x=top50['Liveness']))

# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.66)
fig.show()

