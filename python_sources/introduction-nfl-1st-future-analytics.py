#!/usr/bin/env python
# coding: utf-8

# ### NFL 1st AND FUTURE ANALYTICS - 2019

# In[ ]:


import os
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import chart_studio.plotly as py
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[ ]:


playlist = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')
injuries = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')
tracking = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv')


# In[ ]:


print("Playlist ", playlist.shape)
print("Injury Record ", injuries.shape)
print("Player Track Data ", tracking.shape)


# In[ ]:


print(playlist.columns)
print(injuries.columns)
print(tracking.columns)


# *Playkey is the common column*

# ## Playlist

# In[ ]:


playlist.head()


# In[ ]:


playlist.isnull().any().sum()


# In[ ]:


playlist.dtypes


# In[ ]:


len(np.unique(playlist['PlayKey']))


# **Number of Unique Games Played = 267006**

# In[ ]:


x = playlist['StadiumType'].value_counts().index
y = playlist['StadiumType'].value_counts().values
fig = go.Figure(data=[go.Bar(x=x, y=y,text=y,textposition='auto',marker={'color': 'royalblue'})])
fig.update_xaxes(tickangle=45);fig.update_layout( title={'text': "Stadium Distribution",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Stadium Types",yaxis_title="Number of Games Played",font=dict(family="Courier New, monospace",size=18,color="#7f7f7f"))


# In[ ]:


x = playlist[playlist['Temperature'] != -999]['Temperature'].value_counts().index
y = playlist[playlist['Temperature'] != -999]['Temperature'].value_counts().values
fig = go.Figure(data=[go.Bar(x=x, y=y,text=y,textposition='auto',marker={'color': 'royalblue'})])
fig.update_xaxes(tickangle=45);fig.update_layout( title={'text': "Temprature Distribution",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Temperature (in F)",yaxis_title="Number of Games Played",font=dict(family="Courier New, monospace",size=18,color="#7f7f7f"))


# **Seems that in most of the matches, temprature was not recorded. Hence, it is entered as -999.**

# In[ ]:


frame = { 'Index': playlist[['PlayKey', 'PlayType']].groupby('PlayType').count().sort_values(by='PlayType').index, 'Values': playlist[['PlayKey', 'PlayType']].groupby('PlayType').count()['PlayKey'].values} 
result = pd.DataFrame(frame) 

fig = px.bar(result, x="Index", y="Values", title="Play Type Matches", labels={'Index': 'Types of Play', 'Values': 'Number of Matches'})
fig.show()


# In[ ]:


frame = { 'Index': playlist['FieldType'].value_counts().index, 'Values': playlist['FieldType'].value_counts().values} 
result = pd.DataFrame(frame) 

fig = px.bar(result, x="Index", y="Values", title="Play Type Matches", labels={'Index': 'Turf Type', 'Values': 'Number of Matches'})
fig.show()


# ## Injuries

# In[ ]:


injuries.dtypes


# In[ ]:


injuries.head(3)


# In[ ]:


fig = make_subplots(rows=1, cols=2)
x1 = playlist[['FieldType', 'PlayKey']].drop_duplicates().groupby('FieldType').count().index.values
y1 = playlist[['FieldType', 'PlayKey']].drop_duplicates().groupby('FieldType').count()['PlayKey'].values
x2 = injuries[['Surface', 'PlayKey']].drop_duplicates().groupby('Surface').count().index.values
y2 = injuries[['Surface', 'PlayKey']].drop_duplicates().groupby('Surface').count()['PlayKey'].values
# fig.show()
fig.add_trace(
    go.Bar(x=x1, y=y1, name="Turf Wise Games "),
    row=1, col=1,
)
fig.add_trace(
    go.Bar(x=x2, y=y2, name="Turf Wise Injuries"),
    row=1, col=2
)
fig.update_layout(height=600, width=800, title_text="Plays/Injuries")
fig.show()


# 

# ### More Injuries are observed on Synthetic Turf even though more number of matches were played on Natural Turf

# In[ ]:


df = pd.merge(injuries, playlist, on="PlayKey")


# In[ ]:


df['StadiumType'].value_counts()


# ### More Coming Soon!
