#!/usr/bin/env python
# coding: utf-8

# I will add more soon like predicting the probablity of winning of a team at a particular position
# 
# 
# 
# Any suggestions or comments are appreciated

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_matches = pd.read_csv('../input/matches.csv')


# In[ ]:


df_matches.head()


# In[ ]:


len(df_matches[df_matches.dl_applied>0])


# In[ ]:


len(df_matches)


# In[ ]:


df_matches['winner'].value_counts().plot(kind='bar')


# In[ ]:


df_matches_2008 = df_matches[df_matches['season']==2008]
df_matches_2009 = df_matches[df_matches['season']==2009]
df_matches_2010 = df_matches[df_matches['season']==2010]
df_matches_2011 = df_matches[df_matches['season']==2011]
df_matches_2012 = df_matches[df_matches['season']==2012]
df_matches_2013 = df_matches[df_matches['season']==2013]
df_matches_2014 = df_matches[df_matches['season']==2014]
df_matches_2015 = df_matches[df_matches['season']==2015]
df_matches_2016 = df_matches[df_matches['season']==2016]


# In[ ]:


df_matches_2008['player_of_match'].value_counts().plot(kind='bar',figsize=(15,10))


# In[ ]:


df_matches_2009['player_of_match'].value_counts().plot(kind='bar',figsize=(15,10))


# In[ ]:


#Now some analysis on wins based on the toss


# In[ ]:


df_matches[df_matches['toss_decision']=='bat']['winner'].value_counts().plot(kind='bar',figsize=(15,5))
# Deccan chargers didn't win a single match when they took first batting


# In[ ]:


df_matches[df_matches['toss_decision']!='bat']['winner'].value_counts().plot(kind='bar',figsize=(15,5))


# In[ ]:


# df_matches_2008['toss_decision'].value_counts().values
plt.pie(df_matches['toss_decision'].value_counts().values,
        labels=df_matches['toss_decision'].value_counts().index,
        startangle=90,autopct='%1.1f%%')


# In[ ]:


# Visualising the scenario in which the team won both the toss and match


# In[ ]:


df_matches[df_matches['toss_winner']==df_matches['winner']]['winner'].value_counts().plot(kind='bar',figsize=(15,5))


# In[ ]:


df_matches['city'].value_counts().plot(kind='bar',figsize=(15,5))


# In[ ]:


df_matches['venue'].value_counts().plot(kind='bar',figsize=(15,5))


# In[ ]:


# Analysing the number of wins for a particular team at different Locations

wins_percity = df_matches_2008.groupby(['winner', 'city'])['id'].count().unstack()
plot = wins_percity.plot(kind='bar', stacked=True, title="Team wins in different cities\nSeason "+str(2008), figsize=(15, 5))
sns.set_palette("Paired", len(df_matches_2008['city'].unique()))
plot.set_xlabel("Teams")
plot.set_ylabel("No of wins")
plot.legend(loc='best', prop={'size':8})
#output is showing that playing at hometown gives some advantage may be due to support from fans


# In[ ]:


# we can see that chennai super kings won decent number of matches every season
# There is high variance in the other team wins which played more than one season except pune warriors who performed
# poorly in all the seasons
plot = df_matches.groupby(['winner','season'])['id'].count().unstack().plot(kind='bar',
                                                                     figsize=(15,5),
                                                                     title='Team wins in different seasons')
sns.set_palette("Paired", len(df_matches['id'].unique()))
plot.set_xlabel('Team Name')
plot.set_ylabel('Number of wins')
plot.legend(loc='best')


# In[ ]:


plot =df_matches.groupby(['toss_decision','winner'])['id'].count().unstack().plot(kind='bar',
                                                                                  title='Team win based on toss decision',figsize=(15,5))
plot.set_ylabel('Number of wins')
plot.legend(loc='best')

# we can tell that Royal challengers and Kings XI punjab are good at chasing 
# chennai super kings are good at defending the score rather than chasing
# kkr,MI,RR are better at both chasing and defending


# In[ ]:


## Time for some analysis on ball to ball analysis


# In[ ]:


delivery_df = pd.read_csv('../input/deliveries.csv')


# In[ ]:


delivery_df.head()


# In[ ]:


# delivery_df doesnot contain the mapping from matches to season
# so we add a new column named season to the data frame


# In[ ]:


def season(row):
    if row['match_id']<=58:
        return 2008
    if row['match_id']>=59 and row['match_id']<116:
        return 2009
    if row['match_id']>=116 and row['match_id']<176:
        return 2010
    if row['match_id']>=176 and row['match_id']<249:
        return 2011
    if row['match_id']>=249 and row['match_id']<323:
        return 2012
    if row['match_id']>=323 and row['match_id']<399:
        return 2013
    if row['match_id']>=399 and row['match_id']<459:
        return 2014
    if row['match_id']>=459 and row['match_id']<518:
        return 2015
    if row['match_id']>=518 and row['match_id']<578:
        return 2016


# In[ ]:


delivery_df['season'] = delivery_df.apply(lambda row:season(row),axis=1)


# In[ ]:


# the number of dismisaals due to catches are high in every season since players try for bigger shots due to the 
# less number of overs in the match
delivery_df.groupby('season')['dismissal_kind'].value_counts().unstack().plot(kind='bar',figsize=(15,5))


# In[ ]:


plot = delivery_df.groupby('over')['dismissal_kind'].value_counts().unstack().plot(kind='bar',
                                                                                   figsize=(20,10),
                                                                                   title='No of wickets collapsing in each over')
plot.set_ylabel('No of wickets')
plot.legend(loc='best')
# we can see that there is a gradual decrease in the number of catches after 6th over till 13th over and slowly 
# increases after 15th over
# we can expect this kind of trend due to the different field setting after 6th over because power play 2 starts
# According to ipl rules 
# --> powerplay 1 --> 1 to 6 overs
# --> powerplay 2 --> 7 to 15 overs
# --> powerplay 3 --> 16 to 20 overs
# we can also observe that there is a gradual increase of catches in the last overs since every batsmen tries to hit
# big shots to reach high scores at the end which will also result in higher number of catches


# In[ ]:


plot = delivery_df.groupby('over')['total_runs'].value_counts().unstack().plot(figsize=(20,10),
                                                                                   title='No of different runs in each over')
plot.set_ylabel('No of Runs')
plot.legend(loc='best')
#we can see that number of dot balls and single runs are gradually decreasing in the last 5 overs
#Dot balls decreased gradually from starting to end
# The number of boundaries increased from starting to end


# In[ ]:




