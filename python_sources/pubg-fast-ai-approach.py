#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Introduction
# 
# The dataset seems a good dataset to try the fastai approach. So, basically my aim is to build a model based on random forest classifier as fast as possible and work from there.
# 
# Sidenote : There's  a bunch of codes that I commented out. I followed the fastai lecture and notebook to a tee just for learning purposes. The code that I commented out are not important to get the result and submission. Feel free to try the codes out and uncommented it. 

# In[ ]:


get_ipython().system('pip install fastai==0.7.0')
get_ipython().system('pip install torchtext==0.2.3')


# In[ ]:


from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from IPython.display import display

from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_raw = pd.read_csv('../input/train_V2.csv', low_memory=False)
df_raw_test = pd.read_csv('../input/test_V2.csv', low_memory=False)


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


# In[ ]:


display_all(df_raw.tail())


# In[ ]:


display_all(df_raw.describe(include='all'))


# ## Initial Processing

# Let store Id, groupId, and matchId from the test dataset into an info dataset. This will be later use for submission. We can then drop this 3 fields from both the train and test dataset as it would not help us in building our model.

# In[ ]:


# store test info
df_raw_test_info = df_raw_test[['Id', 'groupId', 'matchId']]


# In[ ]:


df_raw.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)
df_raw_test.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)


# fastai library provide train_cats method which change any columns of strings in a dataframe to a column of categorical values. I also use apply_cats method on the test dataset to change any columns of strings into categorical variables using the train dataset (df_raw) as a template.

# In[ ]:


train_cats(df_raw)
apply_cats(df_raw_test, df_raw)


# In[ ]:


display_all(df_raw.isnull().sum().sort_index()/len(df_raw))


# Apparently, we have na values in the winPlacePerc column. This column is the target variable so it does not make sense to have an na values. Let's check the rows.

# In[ ]:


df_raw[pd.isna(df_raw['winPlacePerc'])]


# Oh, it's only one row. Let's get rid of the row.

# In[ ]:


df_raw.dropna(subset=['winPlacePerc'], inplace=True)


# Random forest classifier only take entirely numeric dataframe. To make sure our dataset is fit to be passed to the classifier, we can use proc_df to make sure the training and test dataset are set to numeric dataframe. 

# In[ ]:


df_trn, y_trn, nas = proc_df(df_raw, 'winPlacePerc')
df_test, _, _ = proc_df(df_raw_test, na_dict=nas)


# In[ ]:


# m = RandomForestRegressor(n_jobs=-1)
# m.fit(df_trn, y_trn)
# m.score(df_trn,y_trn)


# Running the classifier on the whole dataset give us an r^2 score of 0.986. Not bad. However, we don't have a way to see whether our model will do well for other data. The model is expected to do well on the training data.
# 
# Let see if we can have another dataset to compare. Let split the dataset to get a validation dataset. In building model, validation dataset is a good way to make sure we does not overfit to the training dataset.

# In[ ]:


# split the data to train valid
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(df_trn, y_trn, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.metrics import mean_absolute_error

def print_score(m):
    res = [mean_absolute_error(m.predict(X_train), y_train), mean_absolute_error(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


# m = RandomForestRegressor(n_jobs=-1)
# %time m.fit(X_train, y_train)
# print_score(m)


# Now we can see how good our model is doing. The model is posting an r^2 score of 0.9225 for the validation dataset. Not bad for our first try without much feature engineering. I'm going to use this model to get prediction from test dataset.

# ## Speeding things up

# In[ ]:


# df_train, y_train, nas = proc_df(df_raw, 'winPlacePerc', subset=100000, na_dict=nas)
# df_test, _, _ = proc_df(df_raw_test, na_dict=nas)

# X_train, _, y_train, _ = train_test_split(df_train, y_train, test_size=0.2, random_state=42)


# In[ ]:


# m = RandomForestRegressor(n_jobs=-1)
# %time m.fit(X_train, y_train)
# print_score(m)


# ## Single tree

# In[ ]:


# m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
# m.fit(X_train, y_train)
# print_score(m)


# In[ ]:


# # from IPython import IPython
# import graphviz

# draw_tree(m.estimators_[0], df_trn, precision=3)


# In[ ]:


# def draw_tree(t, df, size=10, ratio=0.6, precision=0):
#     """ Draws a representation of a random forest in IPython.

#     Parameters:
#     -----------
#     t: The tree you wish to draw
#     df: The data used to train the tree. This is used to get the names of the features.
#     """
#     s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
#                       special_characters=True, rotate=True, precision=precision)
#     IPython.display.display(graphviz.Source(re.sub('Tree {',
#        f'Tree {{ size={size}; ratio={ratio}', s)))


# In[ ]:


# m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)
# m.fit(X_train, y_train)
# print_score(m)


# Let's see what happens if we create a bigger tree. The training set result looks great! But the validation set is worse than our original model. This is why we need to use bagging of multiple trees to get more generalizable results.

# ## Bagging
# 
# ### Intro to bagging
# 
# To learn about bagging in random forests, let's start with our basic model again.

# In[ ]:


# m = RandomForestRegressor(n_jobs=-1)
# m.fit(X_train, y_train)
# print_score(m)


# We'll grab the predictions for each individual tree, and look at one example.

# In[ ]:


# preds = np.stack([t.predict(X_valid) for t in m.estimators_])
# preds[:,0], np.mean(preds[:,0]), y_valid[0]


# In[ ]:


# preds.shape


# In[ ]:


# plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);


# The shape of this curve suggests that adding more trees isn't going to help us much. Let's check. (Compare this to our original model on a sample)

# In[ ]:


# m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
# m.fit(X_train, y_train)
# print_score(m)


# In[ ]:


# m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
# m.fit(X_train, y_train)
# print_score(m)


# In[ ]:


# m = RandomForestRegressor(n_estimators=80, n_jobs=-1)
# m.fit(X_train, y_train)
# print_score(m)


# ### Out-of-bag (OOB) score
# 
# Is our validation set worse than our training set because we're over-fitting, or because the validation set is for a different time period, or a bit of both? With the existing information we've shown, we can't tell. However, random forests have a very clever trick called out-of-bag (OOB) error which can handle this (and more!)
# 
# The idea is to calculate error on the training set, but only include the trees in the calculation of a row's error where that row was not included in training that tree. This allows us to see whether the model is over-fitting, without needing a separate validation set.
# 
# This also has the benefit of allowing us to see whether our model generalizes, even if we only have a small amount of data so want to avoid separating some out to create a validation set.
# 
# This is as simple as adding one more parameter to our model constructor. We print the OOB error last in our print_score function below.

# In[ ]:


# m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
# m.fit(X_train, y_train)
# print_score(m)


# ## Reducing over-fitting
# 
# ### Subsampling
# 
# It turns out that one of the easiest ways to avoid over-fitting is also one of the best ways to speed up analysis: subsampling. Let's return to using our full dataset, so that we can demonstrate the impact of this technique.

# In[ ]:


from sklearn.model_selection import train_test_split

df_train, y_train, nas = proc_df(df_raw, 'winPlacePerc')
df_test, _, _ = proc_df(df_raw_test, na_dict=nas)

X_train, X_valid, y_train, y_valid = train_test_split(df_train, y_train, test_size=0.2, random_state=42)


# The basic idea is this: rather than limit the total amount of data that our model can access, let's instead limit it to a different random subset per tree. That way, given enough trees, the model can still see all the data, but for each individual tree it'll be just as fast as if we had cut down our dataset as before.

# In[ ]:


set_rf_samples(50000)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# Since each additional tree allows the model to see more data, this approach can make additional trees more useful.

# In[ ]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# ### Tree building parameters
# 
# We revert to using a full bootstrap sample in order to show the impact of other over-fitting avoidance methods.

# In[ ]:


# reset_rf_samples()


# In[ ]:


# def dectree_max_depth(tree):
#     children_left = tree.children_left
#     children_right = tree.children_right

#     def walk(node_id):
#         if (children_left[node_id] != children_right[node_id]):
#             left_max = 1 + walk(children_left[node_id])
#             right_max = 1 + walk(children_right[node_id])
#             return max(left_max, right_max)
#         else: # leaf
#             return 1

#     root_node_id = 0
#     return walk(root_node_id)


# In[ ]:


# m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
# m.fit(X_train, y_train)
# print_score(m)


# In[ ]:


# t=m.estimators_[0].tree_


# In[ ]:


# dectree_max_depth(t)


# In[ ]:


# m = RandomForestRegressor(n_estimators=40, min_samples_leaf=5, n_jobs=-1, oob_score=True)
# m.fit(X_train, y_train)
# print_score(m)


# In[ ]:


# t=m.estimators_[0].tree_


# In[ ]:


# dectree_max_depth(t)


# In[ ]:





# In[ ]:


pred = m.predict(df_test)
pred


# In[ ]:


df_sub = df_raw_test_info[['Id']]


# In[ ]:


df_sub['winPlacePerc'] = pred


# In[ ]:


df_sub.to_csv('PUBG_sub.csv', index=None)


# In[ ]:


df_sub


# In[ ]:




