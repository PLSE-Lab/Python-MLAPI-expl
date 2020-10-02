#!/usr/bin/env python
# coding: utf-8

# ## Random Forest Regressor using fast.ai along with Model Interpretation

# To ensure no module not found errors

# In[ ]:


get_ipython().system('pip install fastai==0.7.0 ')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.imports import *
from fastai.structured import *   ## Need to install fastai 0.7 for this 

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn import metrics

import os
print(os.listdir("../input"))


# In[ ]:


df_raw = pd.read_csv('../input/train_V2.csv', low_memory=False)
df_raw_test = pd.read_csv('../input/test_V2.csv', low_memory=False)


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 100, "display.max_columns", 100):
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


# ## Splitting into Validation set
# Let's split the dataset to get a validation dataset to make sure we do not overfit to the training dataset.
# 

# In[ ]:


# split the data to train valid

X_train, X_valid, y_train, y_valid = train_test_split(df_trn, y_trn, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.metrics import mean_absolute_error

def print_score(m):
    res = [mean_absolute_error(m.predict(X_train), y_train), mean_absolute_error(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# ## Running the model 
# Running the model on the enitre dataset took 7 mins. So we are taking a random sub-sample to be able to iterate faster.

# In[ ]:


set_rf_samples(20000)
m = RandomForestRegressor(n_jobs=-1, n_estimators = 40, min_samples_leaf = 7, min_samples_split = 7)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')


# In[ ]:


get_ipython().run_line_magic('time', 'print_score(m)')


# Now we can see how good our model is doing. The model is getting a R^2 score of ~0.91 for the validation dataset. 

# ## Submission

# In[ ]:


pred = m.predict(df_test)
pred


# In[ ]:


df_sub = df_raw_test_info[['Id']]


# In[ ]:


pd.options.mode.chained_assignment = None  # default='warn' ## TO remove warning due to assignment below
df_sub['winPlacePerc'] = pred


# In[ ]:


df_sub.to_csv('PUBG_RF_tune.csv', index=None)


# ## Feature Selection
# Let's check feature importance for each column

# In[ ]:


fi = rf_feat_importance(m, X_train)
fi[:10]


# So Walk distance is THE most important feature that determines the output of a PUBG game !! Sounds intuitive as well !

# In[ ]:


fi.plot('cols', 'imp', figsize=(10,6), legend=False)


# So not all the columns are useful in the Random Forest model, and we can just drop off the some features without sacrificing in accuracy. Let's see how many of these features do we need.

# In[ ]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30])


# In[ ]:


to_keep = fi[fi.imp>0.005].cols; len(to_keep)


# In[ ]:


df_keep = df_raw[to_keep].copy()
X_train, X_valid, y_train, y_valid = train_test_split(df_keep, y_trn, test_size=0.33, random_state=42)


# In[ ]:


set_rf_samples(20000)
m = RandomForestRegressor(n_jobs=-1, n_estimators = 40, min_samples_leaf = 7, min_samples_split = 7)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')


# In[ ]:


get_ipython().run_line_magic('time', 'print_score(m)')


# We see that the R^2 did not change much after removing the redundant features from our model !!

# In[ ]:




