#!/usr/bin/env python
# coding: utf-8

# Sometimes, our LB score could suffer a lot when the train and test set have different distributions. It would be helpful if we could remove some rows to make the train set more similar to the test set. In other words, we need to classify rows in train which are 'abnormal'  ==> Use a binary classification algorithm.
# 
# 
# 
# 
# Proecess: Combine train and test and set their target values as 1 and 0 seperately. Use the algorithm to predict **the target value of train**.
# - If the predict value is 0, it means the algorithm couldn't tell this one from test, which means it is usable.
# - If the predict value is 1, it means the algorithm finds the difference, indicating we should remove this row.
# 
# 
# ![](https://i.imgur.com/cnQmn0W.jpg)
# editted from https://medium.com/crownsontop/are-you-the-black-sheep-in-a-black-family-5a3747d7dd0c

# # Note
# Don't use this method in real-life projects!!!!  It will cause the model **ovefit** the test dataset!Also, don't use it if public and private LB have differnent data distributions.

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.model_selection import KFold, train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score as auc
import plotly.graph_objects as go


# In[ ]:


train_raw=pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")
test_raw=pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

train = train_raw.drop(['id', 'target'], axis=1)
test = test_raw.drop(['id'], axis=1)
y = train_raw.target

cols = train.columns
train_length = train.shape[0]
test_length = test.shape[0]


# # BaseLine Model
# 
# Basic CatboostClassifer without Feature Engineering

# In[ ]:


# cbc = CatBoostClassifier(iterations = 300, learning_rate = 0.1, eval_metric = 'AUC', verbose = False)

# tr_x, val_x, tr_y, val_y = train_test_split(train, y, test_size = 0.2, shuffle = True, random_state = 10)

# cbc.fit(tr_x, tr_y, eval_set=(val_x, val_y), cat_features=cols)

# y_pred = cbc.predict_proba(test)[:, 1]

# submission = pd.DataFrame({'id': test_raw.id, 'target': y_pred})
# submission.to_csv('submission.csv', index=False)


# Baseline Model Score on LB:
# 
# ![](https://i.imgur.com/C5EqLRS.png)

# # Use adversarial validation on raw train dataset

# In[ ]:


n_splits = 5

kf = KFold(n_splits=n_splits, shuffle=True, random_state=10)

y_pred_ad = np.zeros(train.shape[0]) ### Initialize an array to record the predicted result from adversarial validation


### For all rows from train, no matter it is in tr_x or val_x, its taget value is setted as 1.
### For rows from test, its taget value is setted as 0.
for tr_range, val_range in kf.split(train):
    tr_x = train.loc[tr_range]
    val_x = train.loc[val_range]
    val_y = np.ones(val_x.shape[0]) 
    
    tr_x_combined = pd.concat([tr_x, test], axis = 0)
    tr_y_combined = np.append(np.ones(tr_x.shape[0]), np.zeros(test_length))
    
    cbc = CatBoostClassifier(iterations = 300, learning_rate = 0.1, eval_metric = 'AUC', verbose = False)
    cbc.fit(tr_x_combined, tr_y_combined, eval_set=(val_x, val_y), cat_features = cols)
    y_pred_ad[val_range] = cbc.predict_proba(val_x)[:, 1]  


# I prefer using "predict_proba" because in this case, we could manually select a proper propability to decide how many rows we remove from train and how strict the selection is. It is a tradeoff between **quality** and **quantity**.
# 
# If you are familiar with "the elbow method", you would have an idea of choosing which point.  :)
# 
# Alternatively, you could use "predict" to simplify the steps.

# In[ ]:


lox = []
loy = []
for i in range(50, 60, 1): ### Proba from 0.5 to 0.6
    threshold = i / 100
    lox.append(threshold)
    train_ad = train[y_pred_ad < threshold]
    loy.append(train_ad.shape[0])
fig = go.Figure(data=go.Scatter(x=lox, y=loy))
fig.show()


# In this competition, it seems that there is no big differences between train and test since most predicted probabilities lie in the range [0.5, 0.6], i.e. the model couldn't confidently say whether one training example is normal or not.
# 
# Remove the bad-performance rows and train a CatBoostClassifier of the same hyperparamters in baseline model to compare the performance.

# In[ ]:


train = train[y_pred_ad < 0.56]
y = y[y_pred_ad < 0.56]


# In[ ]:


cbc = CatBoostClassifier(iterations = 300, learning_rate = 0.1, eval_metric = 'AUC', verbose = False)

tr_x, val_x, tr_y, val_y = train_test_split(train, y, test_size = 0.2, shuffle = True, random_state = 10)

cbc.fit(tr_x, tr_y, eval_set=(val_x, val_y), cat_features=cols)

y_pred = cbc.predict_proba(test)[:, 1]

submission = pd.DataFrame({'id': test_raw.id, 'target': y_pred})
submission.to_csv('submission.csv', index=False)


# New Score:
# 
# ![](https://i.imgur.com/e8eMURx.png)
# 
# The score has been imporved from 0.79989 to 0.80099! Considering to the similarity of train and test in this competition which limits this method's performance, it is still a great improvement!

# If you like this kernel, please upvote it. Thanks!
