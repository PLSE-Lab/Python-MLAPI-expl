#!/usr/bin/env python
# coding: utf-8

# # Movie Review Competition Public Leaderboard
# 
# 

# In[ ]:


import numpy as np 
import pandas as pd
import os
print(os.listdir("../input/kuzushiji-competition-leaderboard"))
get_ipython().run_line_magic('matplotlib', 'inline')
#from bubbly.bubbly import bubbleplot 
#from __future__ import division
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[ ]:


data_df = pd.read_csv("../input/kuzushiji-competition-leaderboard/movie-reviews-classification-publicleaderboard.csv")
data_df = data_df.loc[data_df.Score > 0.505]


# In[ ]:


teams = list(data_df.TeamName.value_counts().index)
data = []
for team in teams:
    dT = data_df[data_df['TeamName'] == team]
    trace = go.Scatter(
        x = dT['SubmissionDate'],y = dT['Score'],
        name=team,
        mode = "markers+lines"
    )
    data.append(trace)

layout = dict(title = 'Public Leaderboard Submissions (Score > 0.50)',
          xaxis = dict(title = 'Submission Date', showticklabels=True), 
          yaxis = dict(title = 'Team Score'),
          #hovermode = 'closest'
          height=600
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='public-leaderboard')

