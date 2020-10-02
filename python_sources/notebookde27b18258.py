#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:





# Our plan is to use the tournament seeds as a predictor of tournament performance. 
# We will train a logistic regressor on the difference in seeding between the two teams playing, and have the result of the game as the desired output
# 
# This is inspired by [last years competition][1], where [Jared Cross made a model just based on the team seeds][2].
# 
# 
#   [1]: https://www.kaggle.com/c/march-machine-learning-mania-2016
#   [2]: https://www.kaggle.com/jaredcross/march-machine-learning-mania-2016/getting-started

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Load the training data ##
# We are just having a look at the format of the training data to make sure that it looks like we expect

# In[ ]:


data_dir = '../input/'
df_seeds = pd.read_csv(data_dir + 'TourneySeeds.csv')
df_tour = pd.read_csv(data_dir + 'TourneyCompactResults.csv')


# In[ ]:


df_seeds.head()


# In[ ]:


df_tour.head()


# Let's drop the columns we are not planning on using

# In[ ]:


df_tour.drop(labels=['Daynum', 'Wscore', 'Lscore', 'Wloc', 'Numot'], inplace=True, axis=1)


# Get the seeds as integers

# In[ ]:


def seed_to_int(seed):
    """Get just the digits from the seeding. Return as int"""
    s_int = int(seed[1:3])
    return s_int
df_seeds['n_seed'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label


# ## Merge seed for each team ##
# We want the seeds in the same DataFrame as the game results

# In[ ]:


df_winseeds = df_seeds.rename(columns={'Team':'Wteam', 'n_seed':'win_seed'})
df_lossseeds = df_seeds.rename(columns={'Team':'Lteam', 'n_seed':'loss_seed'})


# In[ ]:


df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'Wteam'])
df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'Lteam'])
df_concat['seed_diff'] = df_concat.win_seed - df_concat.loss_seed


# ## Make a new DF with just the wins and losses ##

# In[ ]:


df_wins = pd.DataFrame()
df_wins['seed_diff'] = df_concat['seed_diff']
df_wins['result'] = 1

df_losses = pd.DataFrame()
df_losses['seed_diff'] = -df_concat['seed_diff']
df_losses['result'] = 0

df_for_predictions = pd.concat((df_wins, df_losses))


# In[ ]:


X_train = df_for_predictions.seed_diff.values.reshape(-1,1)
y_train = df_for_predictions.result.values
X_train, y_train = shuffle(X_train, y_train)


# ## Train the estimator ##
# We use logistic regression, so we have to set a `C` value. We can just try a bunch of different values and then choose the best one.

# In[ ]:


logreg = LogisticRegression()
params = {'C': np.logspace(start=-5, stop=3, num=9)}
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
clf.fit(X_train, y_train)
print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))


# ## Examine the classifier predictions ##

# In[ ]:


X = np.arange(-16, 16).reshape(-1, 1)
preds = clf.predict_proba(X)[:,1]


# In[ ]:


plt.plot(X, preds)
plt.xlabel('Team1 seed - Team2 seed')
plt.ylabel('P(Team1 will win)')


# This looks like we would expect. We are predicting the probability of team1 winning. If that team has a lower seed than team2, there is a high probability of team1 winning.

# ## Get the test data ##

# In[ ]:


df_sample_sub = pd.read_csv(data_dir + 'sample_submission.csv')
n_test_games = len(df_sample_sub)


# In[ ]:


def get_year_t1_t2(id):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in id.split('_'))


# We loop over each row in the `sample_submission.csv` file. For each row, we extract the year and the teams playing. 
# We then look up the seeds for each of those teams *in that season*. 
# Finally we add the seed difference to an array.

# In[ ]:


X_test = np.zeros(shape=(n_test_games, 1))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.id)
    # There absolutely must be a better way of doing this!
    t1_seed = df_seeds[(df_seeds.Team == t1) & (df_seeds.Season == year)].n_seed.values[0]
    t2_seed = df_seeds[(df_seeds.Team == t2) & (df_seeds.Season == year)].n_seed.values[0]
    diff_seed = t1_seed - t2_seed
    X_test[ii, 0] = diff_seed


# ## Make the predictions ##

# In[ ]:


preds = clf.predict_proba(X_test)[:,1]


# In[ ]:


clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.pred = clipped_preds
df_sample_sub.head()


# In[ ]:


df_sample_sub.to_csv('logreg_on_seed.csv', index=False)


# In[ ]:




