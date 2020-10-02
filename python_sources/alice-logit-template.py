#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# a helper function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# In[ ]:


train_df = pd.read_csv('../input/catch-me-if-you-can/train_sessions.csv',
                       index_col='session_id')
test_df = pd.read_csv('../input/catch-me-if-you-can/test_sessions.csv',
                      index_col='session_id')

# Convert time1, ..., time10 columns to datetime type
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# Sort the data by time
train_df = train_df.sort_values(by='time1')

# Look at the first rows of the training set
train_df.head()


# In[ ]:


sites = ['site%s' % i for i in range(1, 11)]
train_df[sites].fillna(0).astype('int').to_csv('train_sessions_text.txt', 
                                               sep=' ', 
                       index=None, header=None)
test_df[sites].fillna(0).astype('int').to_csv('test_sessions_text.txt', 
                                              sep=' ', 
                       index=None, header=None)


# In[ ]:


get_ipython().system('head -5 train_sessions_text.txt')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


cv = CountVectorizer()


# In[ ]:


get_ipython().run_cell_magic('time', '', "with open('train_sessions_text.txt') as inp_train_file:\n    X_train = cv.fit_transform(inp_train_file)\nwith open('test_sessions_text.txt') as inp_test_file:\n    X_test = cv.transform(inp_test_file)\nprint(X_train.shape, X_test.shape)")


# In[ ]:


type(X_train)


# In[ ]:


y_train = train_df['target'].astype(int)


# In[ ]:


y_train.head()


# ### train Logistic regression

# In[ ]:


logit = LogisticRegression(C = 1, random_state=42)


# In[ ]:


get_ipython().run_cell_magic('time', '', "cv_scores = cross_val_score(logit, X_train, y_train, cv= 5, scoring='roc_auc')")


# In[ ]:


cv_scores.mean()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'logit.fit(X_train, y_train)')


# In[ ]:


test_pred_logit1 = logit.predict_proba(X_test)[:,1]


# In[ ]:


write_to_submission_file(test_pred_logit1, 'logit_sub1.txt') ## .908 ROC AUC


# In[ ]:


get_ipython().system('head logit_sub1.txt')


# ### Time Features
# 
# - hour when the session started
# - morning 
# - day
# - eve
# - night

# In[ ]:


def add_time_features(time1_series, X_sparse):
    hour = time1_series.apply(lambda ts: ts.hour)
    morning = ((hour >= 7) & (hour <= 11)).astype('int')
    day = ((hour >= 12) & (hour <= 18)).astype('int')
    evening = ((hour >= 19) & (hour <= 23)).astype('int')
    night = ((hour >= 0) & (hour <= 6)).astype('int')
    X = hstack([X_sparse, morning.values.reshape(-1, 1), 
                day.values.reshape(-1, 1), evening.values.reshape(-1, 1), 
                night.values.reshape(-1, 1)])
    return X


# In[ ]:


test_df.loc[:, 'time1'].fillna(0).apply(lambda ts: ts.hour).head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "X_train_with_time = add_time_features(train_df['time1'].fillna(0), X_train)\nX_test_with_time = add_time_features(test_df['time1'].fillna(0), X_test)")


# In[ ]:


logit_with_time = LogisticRegression(C = 1, random_state=42)


# In[ ]:


get_ipython().run_cell_magic('time', '', "cv_scores = cross_val_score(logit_with_time, X_train_with_time, y_train, cv= 5, scoring='roc_auc');")


# In[ ]:


cv_scores.mean()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'logit_with_time.fit(X_train_with_time, y_train)')


# In[ ]:


test_pred_logit2 = logit_with_time.predict_proba(X_test_with_time)[:,1]


# In[ ]:


write_to_submission_file(test_pred_logit2, 'logit_sub2.txt') ## .93565 ROC AUC


# In[ ]:


get_ipython().system('head logit_sub2.txt')


# In[ ]:




