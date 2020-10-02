#!/usr/bin/env python
# coding: utf-8

#  We have eliminated 7 columns from the original file. We are attempting to streamline the data. There are 666 rows 
# representing every game that was played in 2016 and 2017. We are checking for any correlation. 

# In[ ]:


import pandas as pd


# In[ ]:


import os
print(os.listdir("../input"))


# 42 games have no GameWeather
# 66 have no Temperature
# 256 have no OutdoorWeather
# 

# In[ ]:


df = pd.read_csv('../input/game_data_clean.csv')
df.head()


#  666 rows (333 each year).   18 columns  were  removed because they are not relevant to WHY a punt occurred or what happened as a result of that punt.  This leaves 11 columnsx666rows= 7,326 pieces of data.
#     columns removed 
# Game_Date
# Game_Day
# Start_Time
# Home_Team
# Visit_Team
# Stadium
# Game_Site
# 
