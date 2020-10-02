#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Reading the dataset

# In[ ]:


train_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
test_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
sample_sub_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')


# ## Glimpse of the dataset

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


sample_sub_df.head()


# In[ ]:


print(f'Shape of training dataset: {train_df.shape}')
print(f'Shape of test dataset: {test_df.shape}')


# In[ ]:


train_df.columns


# The `target` is the target variable which we are going to predict for given test dataset.

# ## Handling the missing values (NaNs)

# Let's first check how many missing values are there in training and testing dataset.

# In[ ]:


train_df.isna().sum()


# So there are no records with NaN in target variable and ~18000 NaNs in columns other than `id`

# Let's try to fill NaN using interpolation first. 

# In[ ]:


train_df = train_df.apply(lambda group: group.interpolate(limit_direction='both'))
train_df.isna().sum()


# Okay, so few features now don't have any missing values at all. To fill other NaNs, Let's use mean and mode of the features (Mean for continuous and mode for categorial features). But here we only have categorial features, so we will use mode.

# In[ ]:


for col in train_df:
    if train_df[col].isna().sum() > 0:
        train_df[col] = train_df[col].fillna(train_df[col].mode()[0])


# In[ ]:


train_df.isna().sum()


# Now all the missing values are filled!

# ### Distribution of target variable

# In[ ]:


sns.set(rc={'figure.figsize':(13,8)})
sns.distplot(train_df['target'])
plt.show()


# Let's first divide the data into features and target variable and remove non-features columns

# In[ ]:


Y_train = train_df['target']
X_train = train_df.drop(['target', 'id'], axis=1)
X_test = test_df.drop(['id'], axis=1)


# Now let's convert the entire dataset into one-hot encoding

# In[ ]:


get_ipython().run_cell_magic('time', '', 'combined_data = pd.concat([X_train, X_test], axis=0, sort=False)\ncombined_data = pd.get_dummies(combined_data, columns=combined_data.columns, drop_first=True, sparse=True)\nX_train = combined_data.iloc[: len(train_df)]\nX_test = combined_data.iloc[len(train_df): ]')


# In[ ]:


# Delete the dataframe to decrease memory usage
del train_df
del test_df


# In[ ]:


print(f'Shape of training dataset: {X_train.shape}')
print(f'Shape of test dataset: {X_test.shape}')


# In[ ]:


X_train = X_train.sparse.to_coo().tocsr()
X_test = X_test.sparse.to_coo().tocsr()


# In[ ]:


class ModelHelper(object):
    
    def __init__(self, params, model):
        self.model = model(**params)
    
    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)
    
    def predict(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]
    
    def evaluate(self, Y_true, Y_preds):
        return roc_auc_score(Y_true, Y_preds)


# In[ ]:


SPLITS = 10
kfold = KFold(n_splits=SPLITS, shuffle=False, random_state=666)

lr_params = {
    'verbose': 100,
    'max_iter': 600,
    'C': 0.5,
    'solver': 'lbfgs'
}
model_helper = ModelHelper(lr_params, LogisticRegression)
scores = []
predictions = np.zeros((SPLITS, X_test.shape[0]))
for i, (train_index, test_index) in enumerate(kfold.split(X_train)):
    X_dev, X_val = X_train[train_index], X_train[test_index]
    Y_dev,Y_val = Y_train[train_index], Y_train[test_index]
    
    model_helper.train(X_dev, Y_dev)
    preds = model_helper.predict(X_val)
    roc_score = model_helper.evaluate(Y_val, preds)
    
    full_test_preds = model_helper.predict(X_test)
    scores.append(roc_score)
    predictions[i, :] = full_test_preds

print (scores)


# In[ ]:


sample_sub_df['target'] = predictions[np.argmax(scores)]
sample_sub_df.to_csv('submission.csv', index=False)
sample_sub_df


# In[ ]:




