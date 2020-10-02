#!/usr/bin/env python
# coding: utf-8

# # Social Power NBA Data Analysis

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # What's kind of exploration we can do from the data we have?
# 
# ## Case 1: If I am an agent, what is the range of contract I can negotiate for my player?
# * Player's Salary (Agent takes a percentage of a contract, typically 3-5%.)
# * Player's Endorsement

# ## Data Ingestion 

# In[ ]:


nba_2017_nba_players_with_salary = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_nba_players_with_salary.csv')
nba_2017_endorsements = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_endorsements.csv')
nba_2017_player_wikipedia = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_player_wikipedia.csv')


# In[ ]:


nba_2017_nba_players_with_salary.head(20)


# ## EDA

# In[ ]:


plt.subplots(figsize=(16,16))
ax = plt.axes()
corr = nba_2017_nba_players_with_salary.corr()
sns.heatmap(corr, square = True, xticklabels=corr.columns.values,yticklabels=corr.columns.values, annot = True)


# In[ ]:


nba_2017_nba_players_with_salary.info()


# In[ ]:


nba_2017_nba_players_with_salary.isnull().sum()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

y = nba_2017_nba_players_with_salary['SALARY_MILLIONS']
X = nba_2017_nba_players_with_salary

X = X.drop(['Unnamed: 0', 'Rk', 'PLAYER', '3P%', 'FT%', 'TEAM', 'SALARY_MILLIONS'], axis=1)
X['POSITION'] = X['POSITION'].map({"PG":1, "SG":2, "SF":3, "PF":4, "C":5, "PF-C":6})


# ## Modeling

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, random_state=1)
print(train_X.shape, val_X.shape, train_y.shape, val_y.shape)


# In[ ]:


train_X


# In[ ]:


first_model = RandomForestRegressor(n_estimators=200, max_depth=2, random_state=0).fit(train_X, train_y)

#Returns the coefficient of determination R^2 of the prediction.
first_model.score(train_X, train_y)


# In[ ]:


first_model.score(val_X, val_y)


# > Why the scores of train and test sets are so different when you use diffent max_depth?

# ## What is Permutation Importance ?
# 
# ![](https://github.com/ycheng42/public_images/blob/master/Permutation%20Importance.png?raw=true)

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

# Make a small change to the code below to use in this problem. 
perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)

# uncomment the following line to visualize your results
eli5.show_weights(perm, feature_names = val_X.columns.tolist(), top=100)


# ## Try nonparametric model - K-Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=10, weights='uniform')
knn.fit(X, y) 


# ## Stephen Curry : Original $ 12.11 M

# In[ ]:


print(X[:][8:9])
print('Expected Salary:', knn.predict(X[:][8:9]))


# In[ ]:


knn.kneighbors(X, return_distance=False)


# In[ ]:


neigh =  knn.kneighbors(X, return_distance=False)

for i in neigh[8]:
    print(y[i])


# ## Giannis Antetokounmpo : Original $ 3 M

# In[ ]:


print(X[:][18:19])
print('Expected Salary:', knn.predict(X[:][8:9]))


# In[ ]:


neigh =  knn.kneighbors(X, return_distance=False)

for i in neigh[18]:
    print(y[i])


# ## Conclusions
# 
# * Hyperparameters turning has huge impact on model's accuracy and generalization (under- & over-fitting.) 
# * Based on your business goal, the choice of model is important (interpretability  vs. accuracy). 
# * Collect more/differnt data if needed.

# ## Case 2: If I am a team manager, who is the player I need?
# * Player's Salary
# * Player's Age
# * Probability of Win

# ![](https://github.com/ycheng42/public_images/blob/master/nba_players_tsne.png?raw=true)

# ## This part uses clustering algorithm & tsne to project similar nba players in 2D
# 
# [Embedding Projects](http://projector.tensorflow.org/)

# ## [NFL Big Data Bowl :   $75,000](https://www.kaggle.com/c/nfl-big-data-bowl-2020)
# 
# 
# How many yards will an NFL player gain after receiving a handoff? 

# In[ ]:




