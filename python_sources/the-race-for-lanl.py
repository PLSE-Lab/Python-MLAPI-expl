#!/usr/bin/env python
# coding: utf-8

# # Looking at how teams improved Public LB score over time.
# ** Last Updated June 3, 2019 - The final day of the competition**

# Example of how the plot works (gif not up to date with latest data):
# ![](https://i.imgur.com/DHR598A.gif)

# In[18]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pylab as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
plt.style.use('ggplot')
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]

# Format the data
df = pd.read_csv('../input/lanl-leaderboard/LANL-Earthquake-Prediction-publicleaderboard_06_03_2019.csv')
df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])
df = df.set_index(['TeamName','SubmissionDate'])['Score'].unstack(-1).T
df.columns = [name for name in df.columns]


# # Public LB Scores of Top Teams over time

# In[ ]:


# Interative Plotly
init_notebook_mode(connected=True)
TOP_TEAMS = df.min().loc[df.min() < 1.29].index.values
df_filtered = df[TOP_TEAMS].ffill()
df_filtered = df_filtered.loc[df_filtered.index > '2019-04-21']
# Create a trace
data = []
for col in df_filtered.columns:
    data.append(go.Scatter(
                        x = df_filtered.index,
                        y = df_filtered[col],
                        name=col)
               )
    
iplot(data)


# # All competitors LB Position over Time

# In[ ]:


# Scores of top teams over time
ALL_TEAMS = df.columns.values
df[ALL_TEAMS].ffill().plot(figsize=(20, 10),
                           ylim=(1.0, 1.8),
                           color=color_pal[0],
                           legend=False,
                           alpha=0.01,
                           title='All LANL Teams Scores over Time')
plt.show()


# # Top LB Scores

# In[ ]:


# Create Top Teams List
TOP_TEAMS = df.min().loc[df.min() < 1.35].index.values
df[TOP_TEAMS].min().sort_values().plot(kind='barh',
                                       xlim=(1.0, 1.36),
                                       title='Teams with Scores less than 1.35',
                                       figsize=(12, 15),
                                       color=color_pal[3])
plt.show()


# # Count of LB Submissions that improved score
# ## "Slow and Steady wins the race" ~or~ "Keep overfitting until LB improves"?
# This is the count of times the person submitted and got the fun "You're score improved" notification. This is not the total submission count.

# In[ ]:


df[TOP_TEAMS].nunique().sort_values().plot(kind='barh',
                                           figsize=(12, 15),
                                           color=color_pal[1],
                                           title='Count of Submissions improving LB score by Team')
plt.show()


# # Number of teams by date

# In[ ]:


df.ffill().count(axis=1).plot(figsize=(20, 5), title='Number of Teams in the Competition by Date')
plt.show()

