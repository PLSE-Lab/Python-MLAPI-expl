#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import KFold, cross_val_predict

import lightgbm as lgb


# In[4]:


# Read train and test files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# # Facts
# - As was mentioned in other EDA kernels, current dataset have different distribution in train and test sets.
# - We have more features than records in train set.
# - Some columns are even constant in train set.

# In[5]:


(train_df.iloc[:, 2:].nunique() == test_df.iloc[:, 1:].nunique()).any()


# - As number of unique values in features are different, it is possible that there are no any binary or categorical features (except the case we have some fake data in test set)

# # Assumption
# 
# Lets just pretend that we are working with time series data (e.g. transaction history per day) and try to dig into it from that perspective.

# In[6]:


X_train_orig = train_df.drop(["ID", "target"], axis=1)
X_test_orig = test_df.drop(["ID"], axis=1)

# Apply log transform to target variable
y_train = np.log1p(train_df["target"].values)


# # Investigation
# Checking lgbm performance over initial data

# In[7]:


FOLDS = 10
SEED = 2707
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

model = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=500)


# In[8]:


# For the sake of speed just print the result
# predict = cross_val_predict(model, X_train_orig, y_train, cv=kf)
# print(np.sqrt(np.mean((predict-y_train) ** 2)))

print(1.4794830145766735)


# Then let's create just 6 simple features which are calculated for every row ignoring zeros

# In[9]:


def prepare(data_orig):
    data = pd.DataFrame()
    data['mean'] = data_orig.mean(axis=1)
    data['std'] = data_orig.std(axis=1)
    data['min'] = data_orig.min(axis=1)
    data['max'] = data_orig.max(axis=1)
    data['number_of_different'] = data_orig.nunique(axis=1)               # Number of diferent values in a row.
    data['non_zero_count'] = data_orig.fillna(0).astype(bool).sum(axis=1) # Number of non zero values (e.g. transaction count)
    return data

# Replace 0 with NaN to ignore them.
X_test = prepare(X_test_orig.replace(0, np.nan))
X_train = prepare(X_train_orig.replace(0, np.nan))


# And  immediately check the perfomance of lgbm

# In[10]:


predict = cross_val_predict(model, X_train, y_train, cv=kf)
print(np.sqrt(np.mean((predict-y_train) ** 2)))


# ### 1.385 on 10 folds... no so bad for just 6 features!
# let's investigate them a little bit more

# In[11]:


from pandas.plotting import scatter_matrix

# data = X_train.copy()
# data['target'] = train_df['target']

_ = scatter_matrix(X_train, alpha=0.2, diagonal='kde', figsize=(13, 13))
_ = scatter_matrix(X_test, alpha=0.2, diagonal='kde', figsize=(13, 13))


# ### There are 2 features that looks interesting, lets investigate them closer

# In[12]:


sns.jointplot(x='non_zero_count', y='number_of_different', data=X_train)


# ### Data in the train set has a very strange 'arc'
# Does more 'transactions' = less diversity in 'value' ? What do you think?

# In[13]:


sns.jointplot(x='non_zero_count', y='number_of_different', data=X_test)


# ### Test set on the other hand has straight line. It means that all values in a row are different.
# It may be a sign that most of these values were artificially generated.

# In[14]:


print('in train set:', (X_train['number_of_different'] == X_train['non_zero_count']).sum(), 'out of', X_train.shape[0])
print('in test set', (X_test['number_of_different']==X_test['non_zero_count']).sum(), 'out of', X_test.shape[0])


# ### 27657 of possible fake rows in test set
# Lets check plot without them

# In[15]:


sns.jointplot(x='non_zero_count', y='number_of_different', data=X_test.loc[X_test['number_of_different']!=X_test['non_zero_count']])


# ### Still has difference with train set, but looks much more... natural...
# 
# # Feature importance

# In[16]:


model.fit(X_train, y_train)
gain = model.booster_.feature_importance(importance_type='gain')
gain = 100.0 * gain / gain.sum()
pd.DataFrame(gain, index=model.booster_.feature_name(), columns=['gain']).sort_values('gain', ascending=False)


# ### It also a good idea to take log from features for linear models

# In[19]:


data = X_train.copy()
data['target'] = train_df['target']

_ = scatter_matrix(np.log1p(data), alpha=0.2, diagonal='kde', figsize=(13, 13))


# In[21]:


(data['target'] < data['min']).sum()


# In[22]:


(data['target'] > data['max']).sum()


# ### Target value almost always lays between min and max of other values in the row (should we try LSTM here?)

# In[18]:


_ = scatter_matrix(np.log1p(X_test), alpha=0.2, diagonal='kde', figsize=(13, 13))


# ### That is all for now. Have fun making more conspiracy theories :)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




