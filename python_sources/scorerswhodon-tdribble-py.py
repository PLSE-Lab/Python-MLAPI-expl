#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Run this cell, but please don't change it.

# These lines import the Numpy and Datascience modules.
import numpy as np
from datascience import *

# These lines do some fancy plotting magic
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.simplefilter('ignore', FutureWarning)
from matplotlib import patches
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets




# In[ ]:





# In[ ]:


def identity(array):
    return array 


players = Table.read_table('../input/nba-players-stats-20142015/players_stats.csv')
shots = Table.read_table('../input/nba-shot-logs/shot_logs.csv')
players.apply(lambda x: x.strip(',.'), "Name")
temp = players.apply(lambda x: x.lower(),"Name")
players = players.with_column("name", temp)
players = players.drop("Name")
players = players.move_to_start("name")
players = players.join("name",shots, "player_name")
#departments = players.group("name", identity)
#departments
players
















# In[ ]:


def getDribble(name):
    dribb = players.where("name",name).select("DRIBBLES").where("DRIBBLES", are.not_equal_to("nan"))
    dribbb = sum(dribb.column("DRIBBLES"))/players.where("name", name).num_rows
    return dribbb


    


# In[ ]:





# In[ ]:


#get all names in nba
recordList = []
names = players.column(0)
for name in names:
    if name not in recordList:
        recordList.append(name)
recordList


# In[ ]:


dribbling = []
sums = 0 
threepointp = []
pointss = []


for name in recordList:
    temp = getDribble(name)
    dribbling.append(temp)
for name in recordList:
    temp = players.where("name", name).column("3P%").item(0)
    threepointp.append(temp)
    
for name in recordList:
    temp = players.where("name", name).column("PTS").item(0)
    pointss.append(temp)
print("done")
    
    
    




    

    

    
    


# In[ ]:





# In[ ]:




pts = Table().with_columns("name", recordList, "dribbles", dribbling, "3p%", threepointp, "total_p", pointss)

pts = pts.with_columns("value", pts.column("3p%")/pts.column("dribbles") )
pts.scatter("total_p", "dribbles")

z = pts.where("dribbles", are.below(2)).where("total_p", are.above(1300))
z.show()


# In[ ]:


conclusion: not surprisngly klay thompson is up there, didn't know that jimmy butler took so little dribbles, but this was 2014
    so it might make more sense. lots of big men, i thought there'd be more guards
    And Andrew Wiggins. Never really thought of him as a spot up man but he either wasn't making a lot of his shots or he's better without the ball

