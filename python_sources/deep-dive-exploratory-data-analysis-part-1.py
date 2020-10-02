#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Any results you write to the current directory are saved as output.


# In[ ]:


cd '/kaggle/input/data-science-bowl-2019'


# In[ ]:


# get the data into a pandas dataframe
train = pd.read_csv('train.csv')


# # Task at hand 
# Predicting the learning curve of children from a learning app. 
# Specificly how many attempts would a child take to pass an exam!
# 
# 

# # Lay of the land:
# The data is from a mobile app - measure up -  which aims to imporve quant skills in kids.- pretty noble I say. 
# 
# We can access the online version of the app here : https://measureup.pbskids.org/
# 
# How does it look:  https://drive.google.com/open?id=1auGJpXesLS1GjatwUVrVpwBVwZwNJzhl
# 
# 

# # What data files are we given 
# 
# * Train
# * Test
# * train_labels
# * Specs

# In this first part of the Deep Dive series, we will cover the train data in detail.

# # Overview of column names 
# - **Installation id** refers to one child- the user of the app. Shared devices can create noise
# - **Game id** referes to a session 
# - **Event data**- JSON format of all information about a speicific event. this is helpful in understanding what all happened. Has a lot of information. 
# - **Event count** will track sequence within a game. So one can recreate the actions a user took. Starts from 1 and keeps increasing by 1 for every event. 
# - **Event_code** refers to speicific actions - All the finite distinct actions that can take place are given  by a distinct code. 
# - **World** refers to the various game environments.
# - **Title** gives various activities that can happen in a world. 
# - **Type** Media type of the game or video. Possible values are: 'Game', 'Assessment', 'Activity', 'Clip'.
# - Assessment data is captured in another file called train_label
#     The train label gives us the accuracy group which is what we need to predict.

# # Snapshot of Number and unique values of the train data 

# In[ ]:


no_unique ={}
unique_values = {}
dataframes ={}

for col in train.columns:
    no_unique[col] = train[col].nunique()
    if no_unique[col] <11:
        unique_values[col] = train[col].unique()
    else: 
        unique_values[col] = 'Too many to list.Eg: {}'.format(train[col].unique()[:4])
    
    dataframes[col] = pd.DataFrame([{
         'column_name' : col
     ,   'no_unique_values' : no_unique[col]
    ,   'unique_values' : unique_values[col]
    
    }], index = [col])

df_unique = pd.concat([dataframes[col] for col in train.columns], ignore_index = True)
df_unique.sort_values(by = 'no_unique_values').reset_index()


# # Column name: World
# 
# When we enter the game we have an option to enter one of 3 worlds- 
# * Tree world
# * Magma Peak
# * Crystal Caves
# 
# the data also shows 'None' - This is the opening screen ( the one in the pic) - when no world is chosen.
# 
# Opening Pic: https://drive.google.com/open?id=1H_OZ9MjOMWEsnXKlb_sh2VP-QyqCCjte

# # What can we say about the "worlds"
# 

# # Data distribution of 'worlds"
# 
# * the max number of observations have 'None' which simply corresponds to the starting screen before a world is selected
# * the different worlds are nearly equal in number of observation , however, the max is for Magma Peak... may be the volcono is what kids find attractive :) 

# In[ ]:


g =train.groupby('world')
x = g['installation_id'].nunique()
x.reset_index().plot.bar(x ='world' , y = 'installation_id' )


# # which world is played the most 
# 
# * While the games are nearly equally instaled there is a huge gap in the game time. 
# * Magmapeak is the clear leader in this regard followed by Cystal Caves and a very close third Tree Top city.

# In[ ]:


# # which world is the played the most 
g['game_time'].sum().reset_index().plot.bar(x ='world' , y = 'game_time' )


# # Column name: Type
# 
# Within a world  there can be multiple levels and different Activities , video clips, games, or assessments. Each world comes with a map that outlines what a user can do. 
# 
# - The map gives an idea of the various elements that one can encounter in a specific world. 
# - This example is for Treetop City - Image of Map:   https://drive.google.com/open?id=1Yi0lb-naqFq7H1kkfh4fUeYjaypDH_OG
#     
# The child can choose to do all or one of them. It is not mandatory to follow the sequence. 
# 
# 
# 
# Image of an Activity: An activity or any of the other types show up as a pop-up above the virtual kids in the game:
# https://drive.google.com/open?id=1NIl3fMpul6HzJaoOrNXWJBKta7UGwafK
# 
# Another way to look at Type is that it is a major classification of how time is spent by the child with his/her engagement with the game
# There are 4 of them:
#    * Activity 
#    * Clip
#    * Assessment
#    * Game 

# Before we look deeper into Types let look at the interaction of 'Types' with "Worlds"

# In[ ]:


# # what is the breakup of game and type 
x = train.groupby(['type','world' ])['game_time'].sum().sort_values()

# build a dataframe that has all the types with worlds and corresponding sum of time
type_ = train.type.unique()
df ={}
for t in type_:
    df[t] = (x.get(t).reset_index()).set_index('world').rename(columns = {'game_time':t})
df_f = pd.concat([df[t] for t in type_ ] , ignore_index = False , axis = 1, sort = True)
df_f = df_f.fillna(0)

# get individual series as we want to create a chart with all of the differnt types 
worlds = list(df_f.index)
clip = list(df_f.iloc[:,0])
Activity =  list(df_f.iloc[:,1])
Game =  list(df_f.iloc[:,2])
Assessment = list( list(df_f.iloc[:,3]))
del worlds[2]
del clip[2]
del Activity[2]
del Game[2]
del Assessment[2]

# plot the chart 
plt.xlabel('Worlds')
plt.ylabel('Time')
plt.title('Time spent by each activity in different Worlds')

plt.plot(worlds, clip , label = 'Clip')
plt.plot(worlds, Activity , label = 'Activity')
plt.plot(worlds, Game, label = 'Game' )
plt.plot(worlds, Assessment , label = 'Assessment')
plt.legend(loc = 'best')


# # Time spent on differnt 'Types' in different 'Worlds'
# **Game**
# * Maximum time for any type is on '**Game**' --- this is to be expected -- that is the purpouse right 
# * There seems to an issue with 'Tree-Top-City'. The other two worlds are close together for Game
#     
# **Activity**
# * The **activity** is a close second to the 'Game' -- understanding activites may be very important for us
# * Here Crystal Caves deviates from the others as in the other worlds the time spent between Games and Activity is comparable but not for Crystal Caves.
#     
# **Assessment**
#  
#  * The time for assessment seems to be 4-6% of the overall time spent by the users. 
#  * There seeems to be higher time spent on Tree Top city and the lowest appears to be Magmapeak. It would be interseting to find out why. Are one of the games easier / more difficult than the others. It is important given that the overall data was more for Magmapeak 
# * while we can say nothing now, it will be good to investigate later.
#  
# **Clip**
#  
#  * the data shows 0 for all the worlds. Hmm why is that
#  

# # Number of observations for different Types
# - The graph is a break-down of the number of sessions for each type
# - around 9% of the data is assessment - which looks like a good estimate of what would actualy go on.
# 

# In[ ]:


# how many game sessions were spent on the different types
types_ = list(train.groupby('type')['game_session'].count().sort_values().index)
values = list(train.groupby('type')['game_session'].count().sort_values())
plt.xlabel('Types')
plt.ylabel('Number of Sessions')
plt.title('Breakdown of Sessions by Type')
plt.plot(types_ , values)
 


# # Columns - Event_Count / Event_Code / Game Time

# In[ ]:


# plot the charts

i = 0
plt.figure()

for column in list(train.describe().columns):
    i += 1
    ax_gen  = 'ax'+str(i)
    fig_gen = 'fig'+ str(i)
    fig_gen , ax_gen = plt.subplots()  
    ax_gen.hist(x = train[column].dropna())        
    ax_gen.hist(x = train[column].dropna())
    ax_gen.set_title(column)
#     plt.tight_layout()
#     ax_gen.show()


# **Event_count**
#  - Event count captures how many different actions are taking place in a session. It is a way to track the sequence of events that any child would have taken in any speicific game session. 
#  - As a ballmark if there are too many events may be it is not effective learning. As most of the events are when there is a prompt for not doing the right thing. For example: if i have to arrange 3 things in the right sequence based on height of the items. If i arrange it in 3 tries it is the best, but if i need some guidance, the prompt would go off and then i would need to rearrange it again...all this time the even_count would keep increasing by 1. 
#  - Most of the games have under 70 event_count. There is a steep fall off from the 70 mark and it keeps decreasing exponentialy -- which is to be expected
#  
# **Event Code**
# - There are 42 unique codes. 
# - There seems to be some concentration around key codes
# - These codes would need to be looked at with respect to the differnt activities and worlds. 
# 
# **Game Time**
# 
# - There is a step fall off in game time. This is expected as most sessions would end within a reasonsable time. 
# 
# 
#  

# Thank you for taking the time to read this Kernel.
