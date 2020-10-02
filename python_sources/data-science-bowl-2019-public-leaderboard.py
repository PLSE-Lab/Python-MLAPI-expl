#!/usr/bin/env python
# coding: utf-8

# <h1>Data Science Bowl 2019 - Public Leaderboard</h1>
# 

# In[ ]:


import pandas as pd
import os
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
#from bubbly.bubbly import bubbleplot 
#from __future__ import division
import plotly.graph_objs as go
import plotly.figure_factory as ff

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[ ]:


dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f"Updated {dt_string} (GMT)")


# In[ ]:


data_df = pd.read_csv("../input/data-science-bowl-2019-leaderboard/data-science-bowl-2019-publicleaderboard.csv")
sorted_df = data_df.sort_values(by=['Score'], ascending=False)
sorted_selected_df = sorted_df.drop_duplicates(subset=['TeamId'], keep='first')
first_25_teams = sorted_selected_df.head(25).TeamName


# In[ ]:


data = []
for team in first_25_teams:
    dT = data_df[data_df['TeamName'] == team]
    trace = go.Scatter(
        x = dT['SubmissionDate'],y = dT['Score'],
        name=team,
        mode = "markers+lines"
    )
    data.append(trace)

layout = dict(title = 'Public Leaderboard Submissions (current top 25 teams)',
          xaxis = dict(title = 'Submission Date', showticklabels=True), 
          yaxis = dict(title = 'Team Score'),
          #hovermode = 'closest'
         height=800
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='public-leaderboard')

