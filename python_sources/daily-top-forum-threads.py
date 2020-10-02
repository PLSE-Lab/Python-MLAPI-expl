#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import datetime as dt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
sns.set_palette(sns.color_palette('tab20', 20))
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from datetime import date, timedelta


# In[ ]:


class MetaData():
    def __init__(self, path='/kaggle/input/meta-kaggle'):
        self.path = path

    def ForumMessages(self, usecols, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'ForumMessages.csv'), nrows=nrows, usecols=usecols)
        df['PostDate'] = pd.to_datetime(df['PostDate'])
        return df.rename(columns={'Id': 'ForumMessageId'})

    def ForumMessageVotes(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'ForumMessageVotes.csv'), nrows=nrows)
        df['VoteDate'] = pd.to_datetime(df['VoteDate'])
        return df

    def Forums(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'Forums.csv'), nrows=nrows).rename(columns={'Id': 'ForumId'})

    def ForumTopics(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'ForumTopics.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'ForumTopicId'})

    def Users(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Users.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'UserId'})
    
    def PerformanceTiers(self):
        df = pd.DataFrame([
            [0, 'Novice', '#5ac995'],
            [1, 'Contributor', '#00BBFF'],
            [2, 'Expert', '#95628f'],
            [3, 'Master', '#f96517'],
            [4, 'GrandMaster', '#dca917'],
            [5, 'KaggleTeam', '#008abb'],
        ], columns=['PerformanceTier', 'PerformanceTierName', 'PerformanceTierColor'])
        return df
    
    def UserAchievements(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'UserAchievements.csv'), nrows=nrows)
    
    def Users(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Users.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'UserId'})

start = dt.datetime.now()

START_DATE = '2016-01-01'
md = MetaData('/kaggle/input/meta-kaggle')


# In[ ]:


fmv = md.ForumMessageVotes()
fm = md.ForumMessages(usecols=['Id', 'ForumTopicId', 'PostUserId', 'PostDate'])
ft = md.ForumTopics()

message_upvotes = fmv.groupby(['ForumMessageId', 'VoteDate']).size().reset_index()
message_upvotes.columns = ['ForumMessageId', 'VoteDate', 'Upvotes']
messages = pd.merge(fm, message_upvotes, on='ForumMessageId')
messages.head(2)

daily_topic_votes = messages.groupby(['ForumTopicId', 'VoteDate'])[['Upvotes']].sum().reset_index()
daily_topic_votes['TopicRank'] = daily_topic_votes.groupby('VoteDate')['Upvotes'].rank(ascending=False, method='first')
daily_topic_votes = daily_topic_votes.merge(ft[['ForumTopicId', 'Title']], on='ForumTopicId')
daily_topic_votes = daily_topic_votes.sort_values(by='Upvotes', ascending=False)
daily_topic_votes.head(5)


daily_top_topics = daily_topic_votes[daily_topic_votes.TopicRank == 1]
daily_top_topics = daily_top_topics.sort_values(by='VoteDate', ascending=False)
daily_top_topics.head()
daily_top_topics.Upvotes.sum()
daily_top_topics.shape


# # Daily Top Forum Threads
# 
# There are a few spikes in the weekly total discussion votes.
# These spikes are often a result of a single hot topic.  
# 
# These are the most popular topic categories:
# 
# * **Competition winning solutions**: [1st place with representation learning](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629), 
# [1st place solution overview](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557), etc.
# * **General Kaggle Forum** (e.g. [Kaggle Survey](https://www.kaggle.com/general/36940),
# [Data Scientist Hero](https://www.kaggle.com/general/20388),
# [Kaggle Progression System & Profile Redesign Launch](https://www.kaggle.com/general/22208), etc.
# * **Complaints about extreme competition rules**: [This is insane discrimination](https://www.kaggle.com/c/passenger-screening-algorithm-challenge/discussion/35118),
# [Concerns regarding the competitive spirit](https://www.kaggle.com/c/home-credit-default-risk/discussion/64045), etc.
# * **Leakage of course :)**: [The Data "Property"](https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/61329),
# [The 'Magic' (Leak) feature is attached](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/31870),
# [The Magical Feature](https://www.kaggle.com/c/bosch-production-line-performance/discussion/24065),
# [you were only supposed to blow the * doors off](https://www.kaggle.com/c/talkingdata-mobile-user-demographics/discussion/23286), etc.
# 
# 

# In[ ]:


daily_top_topics = daily_top_topics[daily_top_topics.VoteDate > START_DATE]
data = [
    go.Scatter(
        y=daily_top_topics['Upvotes'].values,
        x=daily_top_topics.VoteDate.astype(str),
        mode='markers',
        marker=dict(sizemode='diameter',
                    sizeref=1,
                    size=np.sqrt(daily_top_topics['Upvotes'].values),
                    color=daily_top_topics['Upvotes'].values,
                    colorscale='Viridis',
                    showscale=True
                    ),
        text=daily_top_topics.Title.values,
    )
]
layout = go.Layout(
    autosize=True,
    title='Daily Hottest Forum Threads',
    hovermode='closest',
    xaxis=dict(title='VoteDate', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of votes (daily)', ticklen=5, gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='dailyTopTopics')


# # 2020
# In 2020 we had two disappointing topics (for very different reasons).
# * [PetFinder.my Contest: 1st Place Winner Disqualified](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/125436)
# * [1st Place Removed Solution - All Faces Are Real Team](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/157983)

# In[ ]:


daily_top_topics = daily_top_topics[daily_top_topics.VoteDate >= '2020-01-01']
data = [
    go.Scatter(
        y=daily_top_topics['Upvotes'].values,
        x=daily_top_topics.VoteDate.astype(str),
        mode='markers',
        marker=dict(sizemode='diameter',
                    sizeref=1,
                    size=np.sqrt(daily_top_topics['Upvotes'].values),
                    color=daily_top_topics['Upvotes'].values,
                    colorscale='Reds',
                    showscale=True
                    ),
        text=daily_top_topics.Title.values,
    )
]
layout = go.Layout(
    autosize=True,
    title='Daily Hottest Forum Threads (2020)',
    hovermode='closest',
    xaxis=dict(title='VoteDate', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of votes (daily)', ticklen=5, gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='dailyTopTopics')


# # Resolution Time

# In[ ]:


users = md.Users().merge(md.PerformanceTiers(), on='PerformanceTier')

upvotes = fmv.groupby(['ForumMessageId']).size().reset_index()
upvotes.columns = ['ForumMessageId', 'Upvotes']
messages = pd.merge(fm, upvotes, on='ForumMessageId')
messages = messages.merge(users[['UserId', 'DisplayName', 'PerformanceTierColor']], left_on='PostUserId', right_on='UserId')
dfdc = messages[messages.ForumTopicId == 157983].sort_values(by='PostDate')
dfdc['n'] = np.arange(len(dfdc))
zillow = messages[messages.ForumTopicId == 45770].sort_values(by='PostDate')
zillow['n'] = np.arange(len(zillow))
dfdc.head()
zillow.head()
zillow.shape, dfdc.shape


# In[ ]:


data = [
    go.Scatter(
        y=dfdc['n'].values,
        x=dfdc.PostDate,
        mode='markers',
        marker=dict(sizemode='diameter',
                    sizeref=0.4,
                    size=np.sqrt(dfdc['Upvotes'].values),
                    color=dfdc.PerformanceTierColor.values,
                    ),
        text=dfdc.DisplayName.values,
    ),
    go.Scatter(
        y=dfdc['n'].values,
        x=dfdc.PostDate,
        mode='lines',
    )
]
layout = go.Layout(
    autosize=True,
    title='Deepfake Detection Challenge - Disqualification Thread',
    hovermode='closest',
    xaxis=dict(title='Time', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Cumulative number of messages', ticklen=5, gridwidth=2, range=[-10, 280]),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='dailyTopTopics')


# In[ ]:


data = [
    go.Scatter(
        y=zillow['n'].values,
        x=zillow.PostDate,
        mode='markers',
        marker=dict(sizemode='diameter',
                    sizeref=0.4,
                    size=np.sqrt(zillow['Upvotes'].values),
                    color=zillow.PerformanceTierColor.values,
                    ),
        text=zillow.DisplayName.values,
    ),
    go.Scatter(
        y=zillow['n'].values,
        x=zillow.PostDate,
        mode='lines',
    )
]
layout = go.Layout(
    autosize=True,
    title='Zillow Prize - Disqualification Thread',
    hovermode='closest',
    xaxis=dict(title='Time', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Cumulative number of messages', ticklen=5, gridwidth=2, range=[-10, 280]),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='dailyTopTopics')


# In[ ]:


end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))

