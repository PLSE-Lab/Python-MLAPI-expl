#!/usr/bin/env python
# coding: utf-8

# I have used [Why Not Logistic Regression](https://www.kaggle.com/peterhurford/why-not-logistic-regression) for experiments with One Hoc Encoding and LogisticRegression model. If you used this notebook you can noticed that step with encoding takes more than 4 minutes. It's too long to play around with features and re-calculate data. So I found way to reduce encoding step time.

# In[ ]:


import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('../input/cat-in-the-dat/train.csv')
test = pd.read_csv('../input/cat-in-the-dat/test.csv')

target = train['target']
train_id = train['id']
test_id = test['id']
train.drop(['target', 'id'], axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

print(train.shape)
print(test.shape)


# The both **STEP 1** and **STEP 2** are an original code from [Why Not Logistic Regression](https://www.kaggle.com/peterhurford/why-not-logistic-regression).

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# STEP 1\ntraintest = pd.concat([train, test])\ndummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)\ntrain_ohe = dummies.iloc[:train.shape[0], :]\ntest_ohe = dummies.iloc[train.shape[0]:, :]\n\nprint(train_ohe.shape)\nprint(test_ohe.shape)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# STEP 2\ntrain_ohe = train_ohe.sparse.to_coo().tocsr()\ntest_ohe = test_ohe.sparse.to_coo().tocsr()')


# **STEP 1** duration time is about *4min*.
# 
# **STEP 2** duration time is about *4s*.
# 
# Let's split **STEP 1** to find part which lasts longer.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# STEP 1.1\ntraintest = pd.concat([train, test])\ndummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)\n\nprint(dummies.shape)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# STEP 1.2\ntrain_ohe = dummies.iloc[:train.shape[0], :]\ntest_ohe = dummies.iloc[train.shape[0]:, :]\n\nprint(train_ohe.shape)\nprint(test_ohe.shape)')


# So we can see that splitting data using *iloc* takes about *4min*.
# 
# What if we cast *dummies* to cst matrix first? Before splitting on *train* and *test* data?

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntraintest = pd.concat([train, test])\ndummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)\n\ndummies_csr = dummies.sparse.to_coo().tocsr()\n\ntrain_ohe = dummies_csr[:train.shape[0], :]\ntest_ohe = dummies_csr[train.shape[0]:, :]\n\nprint(train_ohe.shape)\nprint(test_ohe.shape)')


# Result is less then *20s*.
# 
# So we reduced time of running preparation data step more than **4 minutes**.
# 
# In my opinion it's not bad :)
