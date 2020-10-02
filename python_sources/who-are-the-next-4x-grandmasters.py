#!/usr/bin/env python
# coding: utf-8

# There are 3 Top users for the position ( EDIT : Now 2 for the next two slots - )
# 
# | userid   | Display Name    |
# |----------|-----------------|
# | abhishek | Abhishek Thakur |
# | cdeotte  | Chris Deotte    | 
# | christofhenkel    | Dieter |

# ### How do we get these names ??
# Use only the filter options in pandas to come to this list - 
# Here's the quick and dirty way  - 
# 
# 
# * List all the users who are atleast 'GrandMaster' in Competitions, Discussions, Scripts
# * Take an intersection of all the three sets ( Users who are Grandmaster status in all 3 segments )
# 
# #### Asssumptions 
# * Since the 'DataSets' Grandmaster Achievement Type is unavailable , we assume that the next few 4x GMs come from the group of kagglers who are already 3x GM in Competitions, Discussions, Scripts ( and NOT Dataset ) - I know, this is a wild assumuption - but yeah .
# * #### There is a 'high chance' that the next 4x Grandmaster will come from this intersection set.
# 

# In[ ]:


import numpy as np 
import pandas as pd  


# We need these files 
# * Users.csv  ( Useful Columns =  Id |	UserName | 	DisplayName | 	RegisterDate 	| PerformanceTier )
# * UserAchievements.csv ( Useful Columns =  UserId 	| AchievementType |	Tier)

# In[ ]:


path = "/kaggle/input/meta-kaggle/"
df_user = pd.read_csv(path+"Users.csv")
df_achievements = pd.read_csv(path+"UserAchievements.csv")


# In[ ]:


df_user.head()


# In[ ]:


df_achievements.head()


# In[ ]:


df_achievements.AchievementType.unique()


# Unfortunately, the file does not seem to have the **'DataSets' Grandmaster Achievement Type** - That would have made it not miss out some other contenders.

# ### **List** & Filter the users who are **atleast Tier 4** in each of the three segments ( Competition, Discussion , Script)

# In[ ]:


segments = ['Competitions' ,'Discussion' , 'Scripts' ]
top_ids = {}  # A dictionary of segment-IDs combination

for segment in segments : 
    top_ids[segment] = df_achievements['UserId'][(df_achievements['Tier']>3) & (df_achievements['AchievementType'] == segment) ]
    print("Currently there are {} users who are atleast a {} GrandMaster".format(len(top_ids[segment]) , segment  ))


# ### Take an intersection of the users who are top in all 3 three segments ( Competition, Discussion , Script)

# In[ ]:


top_user_ids = set(top_ids[segments[0]]).intersection(set(top_ids[segments[1]])).intersection(set(top_ids[segments[2]]))


# In[ ]:


print("There are {} users who are Grandmasters in ALL 3 SEGMENTS".format(len(top_user_ids)))


# Their UserIDs  - 

# In[ ]:


top_user_ids


# In[ ]:


df_top_users = df_user[df_user.Id.isin(top_user_ids)]


# In[ ]:


df_top_users[['UserName','DisplayName']]

