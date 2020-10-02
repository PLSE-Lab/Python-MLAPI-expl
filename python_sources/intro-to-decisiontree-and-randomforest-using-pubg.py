#!/usr/bin/env python
# coding: utf-8

# # Introduction to DecisionTree and RandomForest
# 
# In this kernel, we train the PUBG dataset using two broadly used tree-based supervised algorithms namely `DecisionTree` and `RandomForest`.   These algorithms for regression and classification are currently among the most widely used machine learning methods.
# 
# We use the regression models of the mentioned learning algorithms. 
# 
# Note that this kernel has just an introductory purpose and is for giving you some ideas about how the models could work.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Get the Data
# 

# In[ ]:


pubg = pd.read_csv('../input/train_V2.csv')
pubg.head()


# ## Explore the Data

# In[ ]:


pubg.info()


# In[ ]:


# let's see the nOfNull values in each columns
pubg.isnull().sum()


# We only have one missing value in the `winPlacePerc` column.

# In[ ]:


pubg[pubg['winPlacePerc'].isnull()]


# In[ ]:


pubg.matchId.nunique()


# In[ ]:


pubg.matchType.value_counts()


# In[ ]:


pubg.describe()


# In[ ]:


def plot_heatmap(corrmat, title):
    sns.set(style = "white")
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    mask = np.zeros_like(corrmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Draw the heatmap with the mask and correct aspect ratio
    plt.figure(figsize=(20, 20))
    hm = sns.heatmap(corrmat, mask=mask, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, cmap=cmap)
    hm.set_title(title)
    plt.yticks(rotation=0)
    plt.show()


# In[ ]:


cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType']
cols_to_fit = [col for col in pubg.columns if col not in cols_to_drop]
corr = pubg[cols_to_fit].corr()


# In[ ]:


plot_heatmap(corr, "Correlation Table")


# In[ ]:


from pandas.plotting import scatter_matrix

scatter_matrix(pubg[["killPlace", "killPoints", "winPoints", "winPlacePerc"]], figsize=(16, 10));


# ## Prepare the Data

# In[ ]:


pubg_copy = pubg.copy()


# we only one mising value so can fill the missin value with zero since the  `walkDistance` is zero

# In[ ]:


pubg_copy['winPlacePerc'].fillna(0, inplace=True)


# We don't need to scale the data before training a model by using DesicionTree and RandomForest algorithms. In other words, they don't require scaling of the data.

# ## Split the Data

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(pubg_copy, pubg_copy['matchType']):
    strat_train_set = pubg_copy.loc[train_index]
    strat_test_set = pubg_copy.loc[test_index]


# In[ ]:


X_train = strat_train_set[cols_to_fit[:-1]].copy()

y_train = strat_train_set.winPlacePerc.copy()


# In[ ]:


X_test = strat_test_set[cols_to_fit[:-1]].copy()

y_test = strat_test_set.winPlacePerc.copy()


# ### One-Hot-Encoding
# 
# we encode the `matchType` feature using one-hot encoding.

# In[ ]:


X_train_dummy = pd.get_dummies(strat_train_set.matchType)


# In[ ]:


X_test_dummy = pd.get_dummies(strat_test_set.matchType)


# In[ ]:


X_train = pd.concat([X_train, X_train_dummy], axis='columns').copy()


# In[ ]:


X_test = pd.concat([X_test, X_test_dummy], axis='columns').copy()


# ## Train DecisionTreeRegressor Model

# In[ ]:


# DecisionTree
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

dt.fit(X_train, y_train)

dt.score(X_train, y_train)


# In[ ]:


dt.score(X_test, y_test)


# In[ ]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(DecisionTreeRegressor(), X_train, y_train, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

print("Scores:", tree_rmse_scores)
print("Mean:", tree_rmse_scores.mean())
print("Standard deviation:", tree_rmse_scores.std())


# ## Train RandomForestRegressor[](http://) Model

# In[ ]:


# RandomForest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=20)

rf.fit(X_train, y_train)

rf.score(X_train, y_train)


# In[ ]:


rf.score(X_test, y_test)


# In[ ]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(RandomForestRegressor(n_estimators=20), X_train, y_train, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)

print("Scores:", forest_rmse_scores)
print("Mean:", forest_rmse_scores.mean())
print("Standard deviation:", forest_rmse_scores.std())


# In[ ]:




