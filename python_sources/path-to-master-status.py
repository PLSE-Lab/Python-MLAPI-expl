#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# **NOTE:** See [Forum post](https://www.kaggle.com/c/meta-kaggle/forums/t/17131/missing-points-and-competitions/97127#post97127)
# about competitions allowing anonymous entries excluded from the data set.  This analysis does not compensate for this.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_context("poster")


# In[ ]:


teams = pd.read_csv('../input/Teams.csv').rename(columns={'Id': 'TeamId'})
competitions = pd.read_csv('../input/Competitions.csv').rename(columns={'Id': 'CompetitionId'})
users = pd.read_csv('../input/Users.csv').rename(columns={'Id': 'UserId'})
team_memberships = pd.read_csv('../input/TeamMemberships.csv').drop('Id', 1)


# In[ ]:


team_memberships['TeamSize'] = team_memberships.groupby('TeamId')['UserId'].transform(lambda x: x.count())


# In[ ]:


team_competitions = teams[['TeamId', 'TeamName', 'CompetitionId', 'Ranking']].merge(
                        competitions[['CompetitionId', 'CompetitionName', 'DateEnabled',
                                      'Deadline', 'CanQualifyTalent']],
                        how='inner', on='CompetitionId')


# In[ ]:


team_competitions['CompetitionTeamCount'] = team_competitions.groupby('CompetitionId')['TeamId'].transform(lambda x: x.count())
team_competitions['RankingPercent'] = team_competitions.eval('Ranking / CompetitionTeamCount')


# In[ ]:


user_competitions = users[users.Tier == 10][['UserId', 'UserName', 'DisplayName']].merge(
                        team_memberships, how='inner', on='UserId').merge(
                        team_competitions[team_competitions.CanQualifyTalent], how='inner', on='TeamId')


# In[ ]:


user_competitions['FinishTop10'] = user_competitions.Ranking.apply(lambda x: 1 if x <= 10 else 0)
user_competitions['FinishTop10Percent'] = user_competitions.RankingPercent.apply(lambda x: 1 if x <= 0.10 else 0)


# In[ ]:


user_competitions_before_master = []
for UserId, df in user_competitions.groupby('UserId'):
    df = df.sort_values(by='Deadline').reset_index(drop=True)
    # Skip users with no top10 finishes (maybe happened on competition excluded from data)
    if df.FinishTop10Percent.sum() < 2:
        continue
    # shift needed to include competition that made master status
    # need at least 1 top10 and 2 top10 percent to make master
    df_before_master = df[~((df.FinishTop10.cumsum().shift(1) > 0) &
                            (df.FinishTop10Percent.cumsum().shift(1) > 1))]
    user_competitions_before_master.append(df_before_master)
user_competitions_before_master = pd.concat(user_competitions_before_master, ignore_index=True)


# In[ ]:


user_competitions_before_master.groupby('UserId').size().describe()


# In[ ]:


user_competitions_before_master.groupby('UserId').size().hist(bins=30)
plt.title('How Many Competitions To Make Master Status')
plt.ylabel('# of Competitors')
_ = plt.xlabel('Talent Qualifying Competitions')


# In[ ]:


time_to_master = user_competitions_before_master.groupby('UserId').agg({'DateEnabled': 'min', 'Deadline': 'max'})
months_to_master = ((pd.to_datetime(time_to_master.Deadline) - pd.to_datetime(time_to_master.DateEnabled)) / np.timedelta64(1,'M'))
months_to_master.describe()


# In[ ]:


months_to_master.hist(bins=50)
plt.xlabel('Months to Master')
plt.ylabel('# of Competitors')
plt.title('How Many Months To Make Master Status')
_ = plt.xticks(np.arange(0, months_to_master.max(), 6.0))


# In[ ]:


when_made_master = user_competitions_before_master.groupby('UserId')[['Deadline']].max()
when_made_master.Deadline = pd.to_datetime(when_made_master.Deadline)
when_made_master['Count'] = 1
when_made_master = when_made_master.set_index('Deadline')


# In[ ]:


when_made_master.resample('1M', how='sum').Count.fillna(0).cumsum().plot()
plt.ylabel('Cumulative Master Competitors')
plt.xlabel('')
_ = plt.title('Cumulative Master Competitor Counts')

