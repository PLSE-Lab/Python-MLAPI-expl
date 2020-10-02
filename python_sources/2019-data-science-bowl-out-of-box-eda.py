#!/usr/bin/env python
# coding: utf-8

# # Core Idea
# 
# Competition Goal is "how young children learn" so let's observe "how young children spend time on this app". Remember the 10,000 hrs of rule? The more time you spend, more you learn. I'm not using fancy charts because they seems to me destructive to get the important facts. 

# #### Important Note
# 
# * I renamed installation_id to kid_id :), Assuming that each installation_id is belong to a kid

# ## Load Libraries

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = '../input/data-science-bowl-2019/'


# ## Load Data

# In[ ]:


train_df = pd.read_csv(DATA_PATH + 'train.csv')
labels_df = pd.read_csv(DATA_PATH + 'train_labels.csv')


# ## Rename Column

# In[ ]:


train_df = train_df.rename(columns={"installation_id": "kid_id"})
labels_df = labels_df.rename(columns={"installation_id": "kid_id"})


# ## Remove Kids 
# 
# For some kids, assessment results are not exists in train_labels.csv so these kid's sessions can't be used in training process. let's remove them.

# In[ ]:


train_df = train_df[train_df.kid_id.isin(labels_df.kid_id.unique())]


# In[ ]:


train_df.info()


# In[ ]:


train_df.head()


# In[ ]:


labels_df.info()


# In[ ]:


labels_df.head()


# # Calculate Session Duration
# 
# Training dataset has total 175467 unique game sessions so let's calculate session duration and find interesting facts. This is time consuming step.

# In[ ]:


sessions_time = train_df.groupby('game_session').agg({'timestamp': ['min', 'max'],'type' : "unique",'world': "unique",'title': "unique"})
sessions_time.columns = ['Start Time', 'End Time','Type','World','Title']
sessions_time["Duration"]= pd.to_datetime(sessions_time["End Time"]) - pd.to_datetime(sessions_time["Start Time"])
sessions_time["Duration"] = sessions_time["Duration"].apply(lambda x: round(x.total_seconds()/60))
sessions_time["Type"] = sessions_time["Type"].apply(', '.join)
sessions_time["World"] = sessions_time["World"].apply(', '.join)
sessions_time["Title"] = sessions_time["Title"].apply(', '.join)
sessions_time = sessions_time.sort_values('Duration',ascending=False)


# In[ ]:


sessions_time.head(20)


# ## Favorite Types

# In[ ]:


sessions_time.groupby('Type')['Duration'].sum()     .plot(kind='bar', figsize=(15, 5), title='Time Spent on Session Type',colormap='winter')
plt.ylabel("Time (Minutes)")
plt.show()


# ### Conclusion - So kids like to play games (My Kids also :)), and skip clips.

# ## Favorite World

# In[ ]:


sessions_time.groupby('World')['Duration'].sum()     .plot(kind='bar', figsize=(15, 5), title='Time Spent on World',colormap='winter')
plt.ylabel("Time (Minutes)")
plt.show()


# ### Conclusion - Hills are favorite for kids. Spend almost equal time in caves and city. 

# ## Favorite Activities

# In[ ]:


sessions_time[sessions_time["Type"] == 'Activity'].groupby('Title')['Duration'].sum()     .plot(kind='bar', figsize=(15, 5), title='Time Spent on Activities',colormap='winter')
plt.ylabel("Time (Minutes)")
plt.xlabel("Activity")
plt.show()


# ### Conclusion - Bottle Filler is most favorite acitivity in Kids

# ## Favorite Games

# In[ ]:


sessions_time[sessions_time["Type"] == 'Game'].groupby('Title')['Duration'].sum()     .plot(kind='bar', figsize=(15, 5), title='Time Spent on Games',colormap='winter')
plt.ylabel("Time (Minutes)")
plt.xlabel("Game")
plt.show()


# ### Conclusion - ChowTime & Scrub-A-Dub are favorite games

# ## Most Clicked Clips
# 
# Kids don't spend much time on clips So we're finding most views clips.

# In[ ]:


sessions_time[sessions_time["Type"] == 'Clip'].groupby('Title')['Duration'].count()     .plot(kind='bar', figsize=(15, 5), title='Clips Views',colormap='winter')
plt.ylabel("Views")
plt.xlabel("Clip")
plt.show()


# # Suspecious Sessions
# 
# I calculated sessions duration in minutes and note that few duration are exceptionally very high and some have zero duration.

# ## High Duration Sessions

# In[ ]:


# Session duration is more than 10 hrs.
sessions_time[sessions_time["Duration"] > 10*60]


# ### Conclusion - Can we say these kids are mobile addicted?

# ## Zero Duration Sessions
# 
# Kids are kids. They skip the things they don't like. So I think these small sessions can't be used for training as well. 

# In[ ]:


sessions_time[sessions_time["Duration"] <= 0]


# ## Valueable Sessions

# In[ ]:


valueable_sessions = sessions_time[(sessions_time["Duration"] > 0) & (sessions_time["Duration"] < 600)]
sessions_df = train_df[train_df.game_session.isin(valueable_sessions.index)]
labels_df = labels_df[labels_df.game_session.isin(valueable_sessions.index)]


# # Kids Performance

# In[ ]:


kids_performance = pd.DataFrame(labels_df.groupby(['kid_id','accuracy_group'])['num_correct'].count().sort_values()).reset_index().pivot(index='kid_id', columns='accuracy_group',values='num_correct').fillna(0).astype('int32')
kids_performance.columns = ['Failed','> 3rd Attempt','2nd Attempt','1st Attempt']
kids_performance = kids_performance[['1st Attempt','2nd Attempt','> 3rd Attempt','Failed']]
kids_performance


# ## Count Sessions Type

# In[ ]:


sessions = pd.DataFrame(sessions_df.groupby('kid_id')['game_session'].unique())
sessions['game_session'] = sessions['game_session'].apply(lambda x: len(x))
kids_performance = pd.merge(right=kids_performance,left=sessions,left_index=True, right_index=True).sort_values(by='game_session',ascending=False)

sessions_type = pd.DataFrame(sessions_df.groupby(["kid_id","type"])['game_session'].unique())
sessions_type['game_session'] = sessions_type['game_session'].apply(lambda x: len(x))
sessions_type = sessions_type.reset_index().pivot(index='kid_id', columns='type',values='game_session').fillna(0).astype('int32')
kids_performance = pd.merge(right=kids_performance,left=sessions_type,left_index=True, right_index=True).sort_values(by='game_session',ascending=False)
kids_performance = kids_performance.rename(columns={"game_session": "Total Sessions"})


# **Note** - All assessments results are not exists in train_labels.csv so you'll see difference in Total Assessments and sum of 1st Attemp, 2nd Attemp, >3 Attempt and Failed. 

# ## 50 Most Active Kids

# In[ ]:


kids_performance.head(50)


# ### Conclusion - Most of active kids are keen to solve assessments except the f6715a6b, He likes games :)

# ## 50 Least Active Kids

# In[ ]:


kids_performance.tail(50)


# ### Conclusion - These kids directly jump into assessments and never come back

# # Assessments Study
# 
# Let's consider those kids who attempts at least 5 assessments. 

# In[ ]:


kids_performance = kids_performance[kids_performance["Assessment"] > 4]
kids_performance


# ## Success Rate

# In[ ]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
types = ['1st Attempt', '2nd Attempt', '>3rd Attemp', 'Failed']
types_sessions = [np.sum(kids_performance['1st Attempt']),np.sum(kids_performance['2nd Attempt']),np.sum(kids_performance['> 3rd Attempt']),np.sum(kids_performance['Failed'])]
ax.pie(types_sessions, labels = types,autopct='%1.2f%%')
plt.title('Assessments Performance')
plt.show()


# ### Conclusion - Failed rate is very high and 40% assessments solved in first attempt.

# In[ ]:


first_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 3].game_session)]
second_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 2].game_session)]
third_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 1].game_session)]
failed_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 0].game_session)]


# ## Solved in First Attempt

# In[ ]:


first_attempt.groupby('Title')['Duration'].count().sort_values(ascending=False)     .plot(kind='bar', figsize=(15, 5), title='Solved in 1st Attempt',colormap='winter')
plt.ylabel("Count")
plt.xlabel("Title")
plt.show()


# ## Solved in Second Attempt

# In[ ]:


second_attempt.groupby('Title')['Duration'].count().sort_values(ascending=False)     .plot(kind='bar', figsize=(15, 5), title='Solved in 2nd Attempt',colormap='winter')
plt.ylabel("Count")
plt.xlabel("Title")
plt.show()


# ## Solved in 3rd or more Attempts

# In[ ]:


third_attempt.groupby('Title')['Duration'].count().sort_values(ascending=False)     .plot(kind='bar', figsize=(15, 5), title='Solved in 3rd or more Attempts',colormap='winter')
plt.ylabel("Count")
plt.xlabel("Title")
plt.show()


# ## Never Solved

# In[ ]:


failed_attempt.groupby('Title')['Duration'].count().sort_values(ascending=False)     .plot(kind='bar', figsize=(15, 5), title='Never Solved',colormap='winter')
plt.ylabel("Count")
plt.xlabel("Title")
plt.show()


# ### Conclusion - Mushroom Sorter is most easiest and Chest Sorter & Bird Measurer are very hard assessments for kids

# ## Time Consumed to Solve Assessments

# In[ ]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
types = ['1st Attempt', '2nd Attempt', '>3rd Attemp', 'Failed']
types_sessions = [np.sum(first_attempt['Duration']),np.sum(second_attempt['Duration']),np.sum(third_attempt['Duration']),np.sum(failed_attempt['Duration'])]
ax.pie(types_sessions, labels = types,autopct='%1.2f%%')
plt.title('Time Consumed in Assessments')
plt.show()


# ### Conclusion - Kids are working hard to solve hard assessments.

# ## Time Spent in Each Assessment

# In[ ]:


first_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 3].game_session)]
second_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 2].game_session)]
third_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 1].game_session)]
failed_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 0].game_session)]


# In[ ]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
types = list(sessions_time[sessions_time["Type"] == "Assessment"]["Title"].unique())
types_sessions = [np.sum(sessions_time[sessions_time["Title"] == types[0]]['Duration']),np.sum(sessions_time[sessions_time["Title"] == types[1]]['Duration']),np.sum(sessions_time[sessions_time["Title"] == types[2]]['Duration']),np.sum(sessions_time[sessions_time["Title"] == types[3]]['Duration']),np.sum(sessions_time[sessions_time["Title"] == types[4]]['Duration'])]
ax.pie(types_sessions, labels = types,autopct='%1.2f%%')
plt.title('Time Spent in Each Assessment')
plt.show()


# ### Conclusion - Cart Balancer is not interesting for kids. Chest Sorter & Bird Measurer are hard but kids spent good time on these.

# # Final Words
# 
# * This dataset has **suspecious sessions** so you should study them before start training process. 
# * Failed rate is very high. It seems that kids stuck on **Chest Sorter** and **Bird Measurer** assessments.
# * Some kids spend lots of time on this app and try to solve hard assessments dedicatedly.
# * Kids, who attempt assessments without spending time on games or activity, are more likely to give up.
# * Mushroom Sorter & Bottle Filler are most favorite for kids.

# **You can still find out more interesting facts so now its your turn :) Please upvote if this kernel is a worth.**
