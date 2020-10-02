#!/usr/bin/env python
# coding: utf-8

# [](https://www.google.com/url?sa=i&url=http%3A%2F%2Fqbardarien.com%2Fcalendar%2Fq-event-list%2F293-march-madness-2020-at-q-bar-darien%3Fdate%3D2020-03-26-17-00&psig=AOvVaw28oFASgDMYBa6bE2-zqmKC&ust=1584120730706000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCLDAn_-LlugCFQAAAAAdAAAAABAD)

# ## Overview

# This Notebook inspired by the Starter kernel [Basic Starter Kernel - NCAA Women's Dataset](https://www.kaggle.com/juliaelliott/basic-starter-kernel-ncaa-women-s-dataset) by [Jullia Eulliot](https://www.kaggle.com/juliaelliott)
# 
# The idea is to create a basic Logistic Regression Model based on the seed differences between teams !
# Note : The Stage's 1 Submissions File is based on expected outcomes 
# For Stage 2, you will be predicting future outcomes based on the teams selected for the tournament.
# if you need to know more about the competition you can check this [comment](https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/discussion/135057#769427)

# ## Acknowledgements
# 
# #### This Notebook uses such great kernels : 
# 
# - [Logistic Regression on Tournament seeds](https://www.kaggle.com/kplauritzen/notebookde27b18258?scriptVersionId=804590)
# - [2020 March Madness Data - First Look EDA](https://www.kaggle.com/robikscube/2020-march-madness-data-first-look-eda)
# - [NCAAM20no-leak starter](https://www.kaggle.com/code1110/ncaam20-finally-no-leak-starter)
# - [Basic Starter Kernel - NCAA Women's Dataset](https://www.kaggle.com/juliaelliott/basic-starter-kernel-ncaa-women-s-dataset)
# 
# Before we get started I want to thank Kaggle and each member here for sharing such huge concepts , without their efforts i can't write this notebook 
# with such ease !
# 
# So, without wasting any time, let's start with importing some important python modules that I'll be using.
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Loading Data

# In[ ]:


data_dir = '../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/'
df_seeds = pd.read_csv(data_dir + 'MNCAATourneySeeds.csv')
df_tour = pd.read_csv(data_dir + 'MNCAATourneyCompactResults.csv')


# In[ ]:


df_seeds.head()


# In[ ]:


df_tour.head()


# then , Let's drop the columns we are not planning on using
# 
# 

# In[ ]:


def seed_to_int(seed):
    
    """Get just the digits from the seeding. Return as int
    """
    s_int = int(seed[1:3])
    return(s_int)

df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # here the string label


# Let's Check the new dataframe

# In[ ]:


df_seeds.head()


# ## Merge seed for each team
# We want the seeds in the same DataFrame as the game results
# that's why we gonna Merge the Seeds with their corresponding TeamIDs in the compact results dataframe

# In[ ]:


df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})

df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])

df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed # difference seed

# Let's concat dataframes and show it !

df_concat.head()


# Next  we'll create a new  dataframe that summarizes wins & losses along with their corresponding seed differences.
# 
# This is the meat of what we'll be creating our model on.

# In[ ]:


# Winners 

df_wins = pd.DataFrame()
df_wins['SeedDiff'] = df_concat['SeedDiff']
df_wins['result'] = 1

# Losses
df_losses = pd.DataFrame()
df_losses['SeedDiff'] = -df_concat['SeedDiff']
df_losses['result'] = 0

# Concat them together

df_for_predictions = pd.concat((df_wins, df_losses))


# In[ ]:


df_for_predictions.head()


# In[ ]:


X_train = df_for_predictions.SeedDiff.values.reshape(-1,1)

y_train = df_for_predictions.result.values


# Let's Shuffle Our DataFrame

# In[ ]:


X_train, y_train = shuffle(X_train, y_train)


# ## Train Our Linear Regression Model
# We'll Use a basic logistic regression to train Our model. 
# You can set different C values to see how performance changes.
# 
# 
# 

# In[ ]:


logregModel = LogisticRegression()

params = {'C': np.logspace(start=-5, stop=3, num=9)}
clf = GridSearchCV(logregModel, params, scoring='neg_log_loss', refit=True)
clf.fit(X_train, y_train)


# ## Evaluate the Model and show the best score

# In[ ]:


print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))


# ### Examine the classifier predictions

# In[ ]:


X = np.arange(-15, 15).reshape(-1, 1)
preds = clf.predict_proba(X)[:,1]


# In[ ]:


plt.plot(X, preds)
plt.xlabel('Team1 seed - Team2 seed')
plt.ylabel('P(Team1 will win)')


# Note :  as Shown below  the probability a team will win decreases as the seed differential to its opponent decreases as well .

# ## Make predictions

# In[ ]:


df_sample_sub = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

n_test = len(df_sample_sub)

# get the year

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))


# In[ ]:


X_test = np.zeros(shape=(n_test, 1))

for i, row in df_sample_sub.iterrows():
    
    year, t1, t2 = get_year_t1_t2(row.ID)
    
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
    diff_seed = t1_seed - t2_seed
    X_test[i, 0] = diff_seed


# Next , we will make prediction with our LR Model

# In[ ]:


preds = clf.predict_proba(X_test)[:,1] # predictor


clipped_preds = np.clip(preds, 0.05, 0.95) # clipped predictions

df_sample_sub.Pred = clipped_preds


# Show up the result

# In[ ]:


df_sample_sub.head()


# Finally, create your submission file!
# 
# 

# In[ ]:


df_sample_sub.to_csv('Submission.csv', index=False)


# ## I hope you find this kernel useful and enjoyable.
# ## Your comments and feedback are most welcome.
# 
# ## Happy Kaggling

# In[ ]:




