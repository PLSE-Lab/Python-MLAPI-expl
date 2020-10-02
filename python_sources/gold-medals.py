#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(sorted(os.listdir("../input")))

# Any results you write to the current directory are saved as output.


# This notebook was prompted by the discusison on teaming influence on gold medals initiated by @kazanova: [Concerns regarding the competitive spirit (103 people currently at Gold)](https://www.kaggle.com/c/home-credit-default-risk/discussion/64045)

# Let's load some data.

# In[ ]:


teams = pd.read_csv('../input/Teams.csv').rename(columns={'Id':'TeamId'})
teams.head()


# In[ ]:


team_memberships = pd.read_csv('../input/TeamMemberships.csv')
team_memberships.head()


# In[ ]:


users = pd.read_csv('../input/Users.csv').rename(columns={'Id':'UserId'})
users.head()


# Let's compute team size and join with the first data frame

# In[ ]:


tmp_df = team_memberships.groupby('TeamId').UserId.count().to_frame('Size').reset_index()
teams = teams.merge(tmp_df, how='left', on='TeamId')
teams.head()  


# I don't know the mapping from medal color to interges (is gold a 3 or a 1), so let's try to find out from the number of medals awarded so far

# In[ ]:


teams.Medal.value_counts()


# OK, 1 is likely to be Gold.  

# Let's look at Solo gold medals.  

# In[ ]:


solo_teams = teams[(teams.Size == 1) & (teams.Medal == 1)]
solo_teams = solo_teams.merge(team_memberships, how='left', on='TeamId')
solo_teams = solo_teams.merge(users, how='left', on='UserId')
solo_teams.head()


# We can now compute the number of solo gold medals per user

# In[ ]:


solo_gold = solo_teams.groupby(['UserName', 'DisplayName']).Medal.count().sort_values(ascending=False)
solo_gold.head(20)


# 

# We recognize some well known names...

# What about gold medals since (last ranking system change)[http://blog.kaggle.com/2016/07/11/kaggle-progression-system-profile-design-launch/]?  This change happened on 07.11.2016. Let's keep competitions that ended after that date.  A quick and dirty way is to keep submissions made after that date.  Beter would be to merge with competition data and use end of competition date.

# In[ ]:


solo_teams['LastSubDate'] = pd.to_datetime(solo_teams.LastSubmissionDate)
recent_solo_teams = solo_teams[solo_teams.LastSubDate > '2016-07-11']
recent_solo_gold = recent_solo_teams.groupby(['UserName', 'DisplayName']).Medal.count().sort_values(ascending=False)
recent_solo_gold.head(20)


# I'm surprised to be there in rather good place :)
