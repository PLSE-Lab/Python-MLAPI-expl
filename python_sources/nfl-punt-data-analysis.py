#!/usr/bin/env python
# coding: utf-8

# **The Challenge**
# 
# 
# For the 2018 season, the NFL revised their kickoff rules in an effort to reduce the risk of injury during those plays. By examining injury reports, player position and velocity data, and game video, they were able to understand the game-play circumstances that may exacerbate the risk of injury to players.
# 
# This comprehensive review showed that over the course of all games during the 2015-2017 seasons, the kickoff represented only six percent of plays but 12 percent of concussions. Players had approximately four times the risk of concussion on returned kickoffs compared to running or passing plays. The changes to the kickoff rule aim to address the components that posed the most risk, like the use of a two-man wedge.
# 
# Now, the NFL is challenging Kagglers to help them perform the same examination, this time on punt play rules. They have provided data for all punt plays from the 2016 and 2017 NFL seasons that includes player rosters, on-field position data and video data, including the plays in which a player suffered a concussion.
# 
# Your challenge is to propose specific rule modifications (e.g. changes to the initial formation, tackling techniques, blocking rules etc.), supported by data, that may reduce the occurrence of concussions during punt plays. More details on the entry criteria are available in Overview tab > Evaluation.
# 
# **About The NFL**
# 
# 
# The National Football League is America's most popular sports league, comprised of 32 franchises that compete each year to win the Super Bowl, the world's biggest annual sporting event. Founded in 1920, the NFL developed the model for the successful modern sports league, including national and international distribution, extensive revenue sharing, competitive excellence, and strong franchises across the country.
# 
# The NFL is committed to advancing progress in the diagnosis, prevention and treatment of sports-related injuries. The NFL's ongoing health and safety efforts include support for independent medical research and engineering advancements and a commitment to look at anything and everything to protect players and make the game safer, including enhancements to medical protocols and improvements to how our game is taught and played.
# 
# As more is learned, the league evaluates and changes rules to evolve the game and try to improve protections for players. Since 2002 alone, the NFL has made 50 rules changes intended to eliminate potentially dangerous tactics and reduce the risk of injuries.
# 
# For more information about the NFL's health and safety efforts, please visit www.PlaySmartPlaySafe.com.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from plotly.offline  import download_plotlyjs,init_notebook_mode,plot, iplot
import cufflinks as cf
init_notebook_mode(connected = True)
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')

from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as offline
# Squarify for treemaps
import squarify
# Random for well, random stuff
import random
# operator for sorting dictionaries
import operator
# For ignoring warnings
import warnings
warnings.filterwarnings('ignore')



video_review = pd.read_csv('../input/video_review.csv')
video_review.head()


# In[ ]:


video_review.info()


# In[ ]:


video_review.describe()


# In[ ]:


# Player activity during primary injury causing event



temp = video_review["Player_Activity_Derived"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, 1]},
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Player activity during primary injury causing event",
        "annotations": [
            {
                "font": {
                    "size": 17
                },
                "showarrow": False,
                "text": "Player activity",
                "x": 0.5,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')


# In[ ]:


# Primary_Impact_Type


temp = video_review["Primary_Impact_Type"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, 1]},
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Impacting source that caused the concussion",
        "annotations": [
            {
                "font": {
                    "size": 17
                },
                "showarrow": False,
                "text": "Primary Impact type",
                "x": 0.5,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')


# In[ ]:


# Game data

game_data = pd.read_csv('../input/game_data.csv')
game_data.head(2)


# In[ ]:


cnt_srs = game_data['Turf'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
    ),
)

layout = go.Layout(
    title='Turf type'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")


# In[ ]:


cnt_srs = game_data['Start_Time'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
    ),
)

layout = go.Layout(
    title='Start time'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")


# .....to be continued

# In[ ]:




