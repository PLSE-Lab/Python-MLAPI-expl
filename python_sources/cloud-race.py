#!/usr/bin/env python
# coding: utf-8

# ## Let us track the public LB standings of Understanding Clouds from Satellite Images
# Reference: https://www.kaggle.com/robikscube/the-race-to-predict-molecular-properties/data
# 
# Last updated: Nov 16, 2019

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pylab as plt
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.linear_model import LinearRegression
import datetime
import colorlover as cl
plt.style.use('ggplot')
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]


# In[ ]:


# Format the dataa
df = pd.read_csv('../input/cloud-leaderboard-race/understanding_cloud_organization-publicleaderboard-nov16.csv')
df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])
df_tmp = df.set_index(['TeamName','SubmissionDate'])['Score']
df = df_tmp[~df_tmp.index.duplicated()]
df = df.unstack(-1).T
df.columns = [name for name in df.columns]
 
TWELFTH_SCORE = df.max().sort_values(ascending=False)[15]
TOP_SCORE = df.max().sort_values(ascending=False)[0]


# ### Race after Oct 20

# In[ ]:


# Interative Plotly
mypal = cl.scales['9']['div']['Spectral']
colors = cl.interp( mypal, 15 )
annotations = []
init_notebook_mode(connected=True)
TOP_TEAMS = df.max().loc[df.max() > TWELFTH_SCORE].sort_values(ascending=False).index[:12].values
df_filtered = df[TOP_TEAMS].ffill()
df_filtered = df_filtered[df_filtered.index >= pd.to_datetime('2019-10-20')]
team_ordered = df_filtered.loc[df_filtered.index.max()]     .sort_values(ascending=False).index.tolist()

data = []
i = 0
for col in df_filtered[team_ordered].columns:
    data.append(go.Scatter(
                        x = df_filtered.index,
                        y = df_filtered[col],
                        name=col,
                        line=dict(color=colors[i], width=2),)
               )
    i += 1

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Cloud Leaderboard Tracking',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))

layout = go.Layout(yaxis=dict(range=[TOP_SCORE-0.02, TOP_SCORE+0.01]),
                   hovermode='x',
                   plot_bgcolor='white',
                  annotations=annotations,
                  )
fig = go.Figure(data=data, layout=layout)
fig.update_layout(
    legend=go.layout.Legend(
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2,
    )
)

fig.update_layout(legend_orientation="h")
fig.update_layout(template="plotly_white")
#fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')
fig.update_xaxes(showgrid=False)

iplot(fig)


# In[ ]:




