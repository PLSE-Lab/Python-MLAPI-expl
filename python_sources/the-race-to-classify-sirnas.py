#!/usr/bin/env python
# coding: utf-8

# # Lets track the Public LB Standings
# ## For the Recursion Cellular Image Classification
# 
# [Shamelessly stolen](https://www.kaggle.com/robikscube/the-race-to-predict-molecular-properties) from @robikscube
# 
# # All credits go to @robikscube.
# 
# I really liked his presentation style. Since he is not in this competition, I am blatantly stealing his code. 

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

# Format the dataa
df = pd.read_csv('../input/recursion-public-leaderboard/recursion-cellular-image-classification-publicleaderboard.csv')
df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])
df = df.set_index(['TeamName','SubmissionDate'])['Score'].unstack(-1).T
df.columns = [name for name in df.columns]

FIFTEENTH_SCORE = df.max().sort_values(ascending=False)[15]
FIFTYTH_SCORE = df.max().sort_values(ascending=False)[50]
TOP_SCORE = df.max().sort_values(ascending=False)[0]


# # Public LB Scores of Top Teams over time

# In[ ]:


# Interative Plotly
mypal = cl.scales['9']['div']['Spectral']
colors = cl.interp( mypal, 15 )
annotations = []
init_notebook_mode(connected=True)
TOP_TEAMS = df.min().loc[df.max() > FIFTEENTH_SCORE].index.values
df_filtered = df[TOP_TEAMS].ffill()
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
                              text='Recursion Leaderboard Tracking',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))

layout = go.Layout(yaxis=dict(range=[0, TOP_SCORE+0.1]),
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


# # All competitors LB Position over Time

# In[ ]:


# Scores of top teams over time
plt.rcParams["font.size"] = "12"
ALL_TEAMS = df.columns.values
df[ALL_TEAMS].ffill().plot(figsize=(20, 10),
                           color=color_pal[0],
                           legend=False,
                           alpha=0.05,
                           ylim=(0, TOP_SCORE + 0.01),
                           title='All Teams Public LB Scores over Time')
df.ffill().max(axis=1).plot(color=color_pal[1], label='1st Place Public LB', legend=True)
plt.show()


# # Number of teams by Date

# In[ ]:


plt.rcParams["font.size"] = "12"
ax =df.ffill()     .count(axis=1)     .plot(figsize=(20, 8),
          title='Number of Teams in the Competition by Date',
         color=color_pal[5], lw=5)
ax.set_ylabel('Number of Teams')
plt.show()


# # Top LB Scores
# Larger bar is better

# In[ ]:


plt.rcParams["font.size"] = "12"
# Create Top Teams List
TOP_TEAMS = df.max().loc[df.max() > FIFTYTH_SCORE].index.values
df[TOP_TEAMS].max().sort_values(ascending=True).plot(kind='barh',
                                       xlim=(0.8,TOP_SCORE),
                                       title='Top 50 Public LB Teams',
                                       figsize=(12, 15),
                                       color=color_pal[3])
plt.show()


# # Distribution of Scores over time

# In[ ]:


plt.rcParams["font.size"] = "7"
n_days = (datetime.date.today() - datetime.date(2019, 6, 27)).days # Num days of the comp
fig, axes = plt.subplots(n_days, 1, figsize=(15, 10), sharex=True)
plt.subplots_adjust(top=8, bottom=2)
for x in range(n_days):
    date2 = df.loc[df.index.date == datetime.date(2019, 6, 27) + datetime.timedelta(x)].index.min()
    num_teams = len(df.ffill().loc[date2].dropna())
    max_cutoff = df.ffill().loc[date2] < 5
    df.ffill().loc[date2].loc[max_cutoff].plot(kind='hist',
                               bins=50,
                               xlim=(0.0,1.0),
                               ax=axes[x],
                               title='{} ({} Teams)'.format(date2.date().isoformat(),
                                                            num_teams))
    y_axis = axes[x].yaxis
    y_axis.set_label_text('')
    y_axis.label.set_visible(False)

