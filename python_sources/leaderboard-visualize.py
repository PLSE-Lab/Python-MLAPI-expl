#!/usr/bin/env python
# coding: utf-8

# # Just for fun
# 
# #### Check out score trends of the top-n competitors.

# In[ ]:


import pandas as pd

def top_n_pivot(top_n):
    leaderboard_df = pd.read_csv("../input/leaderboard/deepfake-detection-challenge-publicleaderboard.csv")
    top = leaderboard_df.groupby('TeamId').last().sort_values(by='Score').head(top_n)
    leaderboard_df['SubmissionDate'] = leaderboard_df['SubmissionDate'].map(lambda date: date[:10])
    leaderboard_df = leaderboard_df.groupby(['TeamId', 'SubmissionDate']).last().reset_index('SubmissionDate').loc[top.index]
    leaderboard_df = leaderboard_df[leaderboard_df['Score'] < 0.5]
    return leaderboard_df.pivot(index='SubmissionDate', columns='TeamName', values='Score').fillna(method='ffill')


# In[ ]:


import cufflinks as cf
cf.go_offline(connected=True)


# In[ ]:


top_n_pivot(5).iplot()


# In[ ]:


top_n_pivot(15).iplot()


# In[ ]:




