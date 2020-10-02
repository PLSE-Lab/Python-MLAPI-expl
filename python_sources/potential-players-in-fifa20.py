#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Under 23 MIdfielders with great potential**

# In[ ]:


#Data Manipulation
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import seaborn as sns

#datetime
import datetime as dt

#warnings
import warnings
warnings.filterwarnings("ignore")


#plotly
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots


# **Reading Data**

# In[ ]:


fifa=pd.read_csv("../input/fifa-20-complete-player-dataset/players_20.csv")
fifa.head()


# **Converting Date of Birth to Date-Time format**

# In[ ]:


fifa.dob=pd.to_datetime(fifa.dob)


# **Creating a dataframe with potential >85 and overall less than 80 but moree than 70**

# In[ ]:


fifa_potential=fifa[(fifa.potential>85)&(fifa.overall<80)]
fifa_potential_ready=fifa_potential[(fifa_potential.overall<80)&(fifa_potential.overall>70)]


# **Potential Goal Keeper**

# In[ ]:


position="GK"
fifa_potential_ready_GK=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_ready_GK


# In[ ]:


fig=go.Figure(data=[go.Bar(name='overall', x=fifa_potential_ready_GK.short_name, y=fifa_potential_ready_GK.overall, text=fifa_potential_ready_GK.overall, textposition='auto'),
                   go.Bar(name='potential', x=fifa_potential_ready_GK.short_name, y=fifa_potential_ready_GK.potential, text=fifa_potential_ready_GK.potential, textposition='auto')
                   ])
fig.update_layout(title='Top potential GK in FIFA 20',
                   xaxis_title='player name ',
                   yaxis_title='Rating')
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = "black"
fig.data[1].marker.line.width = 1
fig.data[1].marker.line.color = "black"
fig.show()


# **Top Potential Left-back players in FIFA 20**

# In[ ]:


position='LB'
position1='LWB'
fifa_potential_LB=fifa_potential_ready[(fifa.player_positions.str.contains(position))|(fifa.player_positions.str.contains(position1))]
fig=go.Figure(data=[
                    go.Bar(name='overall', x=fifa_potential_LB.short_name, y=fifa_potential_LB.overall, text=fifa_potential_LB.overall, textposition='auto'),
                    go.Bar(name='potential', x=fifa_potential_LB.short_name, y=fifa_potential_LB.potential, text=fifa_potential_LB.potential, textposition='auto')
                   ])
fig.update_layout(title='Top Potential LB in FIFA20', xaxis_title='player name', yaxis_title='rating')
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = "black"
fig.data[1].marker.line.width = 1
fig.data[1].marker.line.color = "black"
fig.show()


# **Top Potentital Players in Right back Position**

# In[ ]:


position="RB"
position1="RWB"
fifa_potential_RB=fifa_potential_ready[(fifa.player_positions.str.contains(position))|(fifa.player_positions.str.contains(position1))]
fig=go.Figure(data=[
                    go.Bar(name='overall', x=fifa_potential_RB.short_name, y=fifa_potential_RB.overall, text=fifa_potential_RB.overall, textposition='auto'),
                    go.Bar(name='potential', x=fifa_potential_RB.short_name, y=fifa_potential_RB.potential, text=fifa_potential_RB.potential, textposition='auto')
                   ])
fig.update_layout(title='Top Potential RB in FIFA20', xaxis_title='player name', yaxis_title='rating')
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = "black"
fig.data[1].marker.line.width = 1
fig.data[1].marker.line.color = "black"
fig.show()


# **Top Potential Centre back in FIFA20**

# In[ ]:


position="CB"
fifa_potential_CB=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fig=go.Figure(data=[
    go.Bar(name='overall', x=fifa_potential_CB.short_name, y=fifa_potential_CB.overall, text=fifa_potential_CB.overall, textposition='auto'),
    go.Bar(name='overall', x=fifa_potential_CB.short_name, y=fifa_potential_CB.potential, text=fifa_potential_CB.potential, textposition='auto')
])
fig.update_layout(title='Top Potential CB in FIFA20', xaxis_title='player name', yaxis_title='rating')
fig.data[0].marker.line.width=1
fig.data[0].marker.line.color="black"
fig.data[1].marker.line.width=1
fig.data[1].marker.line.color="black"
fig.show()


# **Top Potential CM in FIFA20**

# In[ ]:


position="CM"
fifa_potential_CM=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fig=go.Figure(data=[
    go.Bar(name='overall', x=fifa_potential_CM.short_name, y=fifa_potential_CM.overall, text=fifa_potential_CM.overall, textposition='auto'),
    go.Bar(name='overall', x=fifa_potential_CM.short_name, y=fifa_potential_CM.potential, text=fifa_potential_CM.potential, textposition='auto')
])
fig.update_layout(title='Top Potential CM in FIFA20', xaxis_title='player name', yaxis_title='rating')
fig.data[0].marker.line.width=1
fig.data[0].marker.line.color="black"
fig.data[1].marker.line.width=1
fig.data[1].marker.line.color="black"
fig.show()


# **Top Potential CAM position**

# In[ ]:


position="CAM"
fifa_potential_CAM=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fig=go.Figure(data=[
    go.Bar(name='overall', x=fifa_potential_CAM.short_name, y=fifa_potential_CAM.overall, text=fifa_potential_CAM.overall, textposition='auto'),
    go.Bar(name='overall', x=fifa_potential_CAM.short_name, y=fifa_potential_CAM.potential, text=fifa_potential_CAM.potential, textposition='auto')
])
fig.update_layout(title='Top Potential CAM in FIFA20', xaxis_title='player name', yaxis_title='rating')
fig.data[0].marker.line.width=1
fig.data[0].marker.line.color="black"
fig.data[1].marker.line.width=1
fig.data[1].marker.line.color="black"
fig.show()


# **Top Potential CDM players**

# In[ ]:


position="CDM"
fifa_potential_CDM=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fig=go.Figure(data=[
    go.Bar(name='overall', x=fifa_potential_CDM.short_name, y=fifa_potential_CDM.overall, text=fifa_potential_CDM.overall, textposition='auto'),
    go.Bar(name='overall', x=fifa_potential_CDM.short_name, y=fifa_potential_CDM.potential, text=fifa_potential_CDM.potential, textposition='auto')
])
fig.update_layout(title='Top Potential CDM in FIFA20', xaxis_title='player name', yaxis_title='rating')
fig.data[0].marker.line.width=1
fig.data[0].marker.line.color="black"
fig.data[1].marker.line.width=1
fig.data[1].marker.line.color="black"
fig.show()


# **Top Potential LW players in FIFA20**

# In[ ]:


position="LW"
fifa_potential_LW=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fig=go.Figure(data=[
    go.Bar(name='overall', x=fifa_potential_LW.short_name, y=fifa_potential_LW.overall, text=fifa_potential_LW.overall, textposition='auto'),
    go.Bar(name='overall', x=fifa_potential_LW.short_name, y=fifa_potential_LW.potential, text=fifa_potential_LW.potential, textposition='auto')
])
fig.update_layout(title='Top Potential LW in FIFA20', xaxis_title='player name', yaxis_title='rating')
fig.data[0].marker.line.width=1
fig.data[0].marker.line.color="black"
fig.data[1].marker.line.width=1
fig.data[1].marker.line.color="black"
fig.show()


# 

# **Top potential RW in FIFA20**

# In[ ]:


position="RW"
fifa_potential_RW=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fig=go.Figure(data=[
    go.Bar(name='overall', x=fifa_potential_RW.short_name, y=fifa_potential_RW.overall, text=fifa_potential_RW.overall, textposition='auto'),
    go.Bar(name='overall', x=fifa_potential_RW.short_name, y=fifa_potential_RW.potential, text=fifa_potential_RW.potential, textposition='auto')
])
fig.update_layout(title='Top Potential RW in FIFA20', xaxis_title='player name', yaxis_title='rating')
fig.data[0].marker.line.width=1
fig.data[0].marker.line.color="black"
fig.data[1].marker.line.width=1
fig.data[1].marker.line.color="black"
fig.show()


# **Top potetial Striker in FIFA20**

# In[ ]:


position="ST"
fifa_potential_ST=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fig=go.Figure(data=[
    go.Bar(name='overall', x=fifa_potential_ST.short_name, y=fifa_potential_ST.overall, text=fifa_potential_ST.overall, textposition='auto'),
    go.Bar(name='overall', x=fifa_potential_ST.short_name, y=fifa_potential_ST.potential, text=fifa_potential_ST.potential, textposition='auto')
])
fig.update_layout(title='Top Potential ST in FIFA20', xaxis_title='player name', yaxis_title='rating')
fig.data[0].marker.line.width=1
fig.data[0].marker.line.color="black"
fig.data[1].marker.line.width=1
fig.data[1].marker.line.color="black"
fig.show()


# **Top potential CF in FIFA20**

# In[ ]:


position="CF"
fifa_potential_CF=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fig=go.Figure(data=[
    go.Bar(name='overall', x=fifa_potential_CF.short_name, y=fifa_potential_CF.overall, text=fifa_potential_CF.overall, textposition='auto'),
    go.Bar(name='overall', x=fifa_potential_CF.short_name, y=fifa_potential_CF.potential, text=fifa_potential_CF.potential, textposition='auto')
])
fig.update_layout(title='Top Potential CF in FIFA20', xaxis_title='player name', yaxis_title='rating')
fig.data[0].marker.line.width=1
fig.data[0].marker.line.color="black"
fig.data[1].marker.line.width=1
fig.data[1].marker.line.color="black"
fig.show()


# In[ ]:


import pandas as pd
players_15 = pd.read_csv("../input/fifa-20-complete-player-dataset/players_15.csv")
players_16 = pd.read_csv("../input/fifa-20-complete-player-dataset/players_16.csv")
players_17 = pd.read_csv("../input/fifa-20-complete-player-dataset/players_17.csv")
players_18 = pd.read_csv("../input/fifa-20-complete-player-dataset/players_18.csv")
players_19 = pd.read_csv("../input/fifa-20-complete-player-dataset/players_19.csv")
players_20 = pd.read_csv("../input/fifa-20-complete-player-dataset/players_20.csv")
teams_and_leagues = pd.read_csv("../input/fifa-20-complete-player-dataset/teams_and_leagues.csv")

