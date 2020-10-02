#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sqlite3
import seaborn as sns
#import matplotlib.pyplot as plt


# In[ ]:


##get a list of competitions
dfCompetitions = pd.read_csv('../input/Competitions.csv')


# In[ ]:


# read in submissions for the competition I'm interested in 
con = sqlite3.connect('../input/database.sqlite')
dfSubmissions = pd.read_sql_query("""
SELECT DateSubmitted, PrivateScore, PublicScore
FROM Submissions
INNER JOIN Teams ON Submissions.TeamId = Teams.Id
WHERE Teams.CompetitionId = 2496""", con)


# In[ ]:


#convert date column from string to date
dfSubmissions['DateSubmitted'] = pd.to_datetime(dfSubmissions['DateSubmitted'])
#drop time
dfSubmissions['DateSubmitted'] = dfSubmissions['DateSubmitted'].apply(lambda x: x.date()) 


# In[ ]:


dfSubmissionScoreMins = dfSubmissions.groupby('DateSubmitted').min()
for i in range(1,len(dfSubmissionScoreMins)):
    if (np.isnan(dfSubmissionScoreMins['PublicScore'][i])) or (dfSubmissionScoreMins['PrivateScore'][i] > dfSubmissionScoreMins['PrivateScore'][i-1]):
        dfSubmissionScoreMins['PrivateScore'][i] = dfSubmissionScoreMins['PrivateScore'][i-1]
        
    if (np.isnan(dfSubmissionScoreMins['PublicScore'][i])) or (dfSubmissionScoreMins['PublicScore'][i] > dfSubmissionScoreMins['PublicScore'][i-1]):
        dfSubmissionScoreMins['PublicScore'][i] = dfSubmissionScoreMins['PublicScore'][i-1]


# In[ ]:


sns.set_style("darkgrid")
dfSubmissionScoreMins.plot()


# In[ ]:





# In[ ]:




