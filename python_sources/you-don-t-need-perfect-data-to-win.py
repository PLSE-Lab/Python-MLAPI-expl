#!/usr/bin/env python
# coding: utf-8

# # Simulating Perfect Data

# The LogLoss statistic is interesting in that it penalizes predictions regardless of the outcome. The nominal penalty for a prediction of 0.5 is 0.69315. If you're more confident than 50/50, then correct predictions are lower; wrong predictions are higher. So, what if you could predict the "true" probability of every game. What would your LogLoss score be?
# 
# This kernel simulates a large number of tournament sets and calculates overall LogLoss statistics. The average LogLoss for a perfect prediction set is 0.5, but varies from 0.25 to 0.75 for a single tournament (Stage 2) and 0.38 to 0.62 for a four-tournament dataset (Stage 1). Scores less than 0.25 for a single tournament are statistical outliers.
# 
# So, regardless of where you end up in the contest, if your final LogLoss is close to 0.5, you've got bragging rights.
# 

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import scipy.stats as st


# Set num_tourneys to 4 for Stage 1 results, or to 1 for Stage 2.
# Set sigma > 1 for more conservative predictions; < 1 for more aggressive predictions.
# Adjust num_rows to change the number of data points in the results dataframe.
# Set normal_dist to False to go straight to uniformly distributed predictions.
# 
# Note: mu and sigma are used to generate z-scores. Nominal predicted probabilities (sigma=1) will have uniform distributions. Conservative predictions will be more bell shaped, while aggressive predictions will be more "U" shaped.

# In[ ]:


# Parameters:

num_tourneys=4  # number of tournaments
num_games=num_tourneys*67 # total number of games in dataset
mu, sigma = 0, 1 # mean and standard deviation for prediction
num_rows=10000 # length of results table
normal_dist = True # normally distributed predictions if True, uniform predictions if False

# Create the outcome data frame.
df=pd.DataFrame(np.zeros((num_games,4)),columns=['Pred','Random','Outcome','Score'])

# Pred = "perfect" predicted probability, normally distributed
# Random = uniformly distributed random number
# Outcome = 1 if Pred>Random, else 0
# Score = the Log Loss calculation

#create the results data frame to record LogLoss for each iteration.
df_results=pd.DataFrame(np.zeros(num_rows),columns=['LogLoss'])


# In[ ]:


# generate num_rows sets of random numbers, determine outcomes, calculate LogLoss

for row in range(num_rows):
    
    if normal_dist:
        df.Pred=st.norm.cdf(np.random.randn(num_games))
    else:
        df.Pred=np.random.rand(num_games)
    
    df.Random=np.random.rand(num_games)
    df.Outcome=np.where(df.Pred>df.Random,1,0)
    df.Score=df.Outcome*np.log(df.Pred)+(1-df.Outcome)*np.log(1-df.Pred)
    df_results.iloc[row,0]=-df.Score.mean()
    
print(df_results.describe())


# In[ ]:





# In[ ]:





# In[ ]:




