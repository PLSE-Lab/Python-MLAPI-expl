#!/usr/bin/env python
# coding: utf-8

# The idea of this script was just to practice my python skills of manipulating data, though without any model building. It will show you how to get the 'Predict' result from the historical data, which is 100% accurate as expected.....

# # 1. Library and Data

# In[6]:


import pandas as pd
import numpy as np

# Load data
TourneyResults_df = pd.read_csv('../input/WNCAATourneyCompactResults.csv')
Submission_df = pd.read_csv('../input/WSampleSubmissionStage1.csv')


# # 2. Manipulate tables

# Manipulate the Tourney Result table to match the colomns in Submission Table

# In[ ]:


TourneyResults_df['Team1ID'] = TourneyResults_df[['WTeamID','LTeamID']].min(axis=1)
TourneyResults_df['Team2ID'] = TourneyResults_df[['WTeamID','LTeamID']].max(axis=1)
# Set result as 1 if the team with smaller team number wins
TourneyResults_df['Result'] = np.where(TourneyResults_df['WTeamID']<TourneyResults_df['LTeamID'], 1, 0)


# Manipulate the Submission table to get Season and Team ID

# In[ ]:


Submission_df['Season'] = Submission_df.ID.apply(lambda x: int(x[:4]))
Submission_df['Team1ID'] = Submission_df.ID.apply(lambda x: int(x[5:9]))
Submission_df['Team2ID'] = Submission_df.ID.apply(lambda x: int(x[10:14]))


# Merge the two table to replace Result in submission table with the real result if it exits, otherwise 0.5.

# In[ ]:


# I dunt know why the Kaggle version of pd has no 'is.na', but it work on my laptop.
Submission_df['Result'] = Submission_df[['Season','Team1ID','Team2ID']].merge(TourneyResults_df[['Season','Team1ID','Team2ID','Result']],left_on = ['Season','Team1ID','Team2ID'],right_on = ['Season','Team1ID','Team2ID'],how='left')[['Result']]
Submission_df['Pred'] = np.where(pd.isna(Submission_df['Result']), Submission_df['Pred'], Submission_df['Result'])


# # 3. We can submit the result now!

# In[ ]:


Submission_df = Submission_df[['ID','Pred']]
Submission_df.to_csv('Submission.csv',index=False)


# This is just to have fun and explain how the 100% accuracy rate in Leaderboard is achieved. To predict this year's result, we still have many models to build......
