#!/usr/bin/env python
# coding: utf-8

# # Additional features and Logit

# In[ ]:


#import libraries
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.sparse import hstack, vstack

from __future__ import division, print_function
# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#column names for sites and time features
site_columns = ['site' + str(i) for i in range(1,11)]
time_columns = ['time' + str(i) for i in range(1,11)]


# In[ ]:


#load train, test data and site dict
train_df = pd.read_csv('../input/train_sessions.csv',
                       index_col='session_id')
test_df = pd.read_csv('../input/test_sessions.csv',
                      index_col='session_id')

train_test_df = pd.concat([train_df, test_df])
train_test_df[site_columns] = train_test_df[site_columns].fillna(0)

with open('../input/site_dic.pkl', 'rb') as f:
    site_dict = pickle.load(f)


# In[ ]:


#function for count-encoding for sites, creating sparse matrix
def to_sparse_matrix(X, site_dict):
    M = max(site_dict.values())
    row_ind = []
    col_ind = []
    data = []
    i, j = 0, 0
    for line in X.values:
        for e in line:
            if e != 0:
                data.append(1)
                row_ind.append(i)
                col_ind.append(e-1)
        i += 1
    return csr_matrix((data, (row_ind, col_ind)), shape=(i, M))


# In[ ]:


#do encoding
train_test_sparse = to_sparse_matrix(train_test_df[site_columns], site_dict)
train_test_sparse.shape


# # Cross-validation for site features and LogisticRegression:

# In[ ]:


X_sparse_train = train_test_sparse[:train_df.shape[0]]
X_sparse_test = train_test_sparse[train_df.shape[0]:]
y_train = train_df.target


# In[ ]:


#stratification for cross-validation
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
logit = LogisticRegression(n_jobs=-1, random_state=42)

score = cross_val_score(estimator=logit, X=X_sparse_train, y=y_train, cv=skf, scoring='roc_auc').mean()
print('ROC AUC score: ', round(score, 4))


# # Create additional time and site features:

# #start day - day of week when session starts
# #start hour - hour when session starts
# #start month - month when session starts
# 
# #stop day - see above
# #stop hour - see above
# #stop month - see above
# 
# #yb start - indicate if session starts with youtube
# #yb stop - indicate if session stops with youtube
# #fb start - indicate if session starts with facebook
# #fb stop - indicate if session stops with facebook
# 
# #duration - how long does session last

# In[ ]:


facebook_ids = []
youtube_ids = []

for key in list(site_dict.keys()):
    if 'facebook' in key:
        facebook_ids.append(site_dict[key])
    if 'youtube' in key:
        youtube_ids.append(site_dict[key])
print(youtube_ids)


# In[ ]:


def is_site(x, l):
    if x in l:
      return 1 
    return 0

def is_long_session(x):
    if x < 3:
        return 0
    elif x < 5:
        return 1
    elif x < 10:
        return 2
    elif x < 30:
        return 3
    elif x < 40:
        return 4
    return 5


# In[ ]:


X_add = train_test_df[['time1']]
X_add['time1'] = train_test_df[['time1']].apply(pd.to_datetime)
X_add['time10'] = train_test_df[['time10']].fillna('2014-02-20 10:02:45').apply(pd.to_datetime)

X_add['start day'] = X_add['time1'].apply(pd.datetime.weekday)
X_add['start hour'] = X_add['time1'].apply(pd.to_datetime).apply(lambda x: x.hour)
X_add['start month'] = X_add['time1'].apply(lambda x: x.month)

X_add['stop day'] = X_add['time10'].apply(pd.datetime.weekday)
X_add['stop hour'] = X_add['time10'].apply(pd.to_datetime).apply(lambda x: x.hour)
X_add['stop month'] = X_add['time10'].apply(lambda x: x.month)

X_add['yb start'] = train_test_df['site1'].apply(lambda x: is_site(x, youtube_ids))
X_add['fb start'] = train_test_df['site1'].apply(lambda x: is_site(x, facebook_ids))

X_add['yb end'] = train_test_df['site10'].apply(lambda x: is_site(x, youtube_ids))
X_add['fb end'] = train_test_df['site10'].apply(lambda x: is_site(x, facebook_ids))

X_add['duration'] = (X_add['time10'] - X_add['time1']).astype(int).apply(lambda x: x/10e8)

X_add = X_add.drop(columns=['time1', 'time10'])

X_add['duration'] = X_add['duration'].apply(is_long_session).astype(int)


# In[ ]:


#dummy encoding for additional features
X_add = pd.get_dummies(X_add, columns=X_add.columns)


# In[ ]:


X_add.columns


# # Cross-validation with site and additional features, Logistic Regression:

# In[ ]:


#merge sets of features
X_train = hstack([X_sparse_train, X_add[:train_df.shape[0]]])
X_test = hstack([X_sparse_test, X_add[train_df.shape[0]:]])


# In[ ]:


logit = LogisticRegression(n_jobs=-1, random_state=42)


# In[ ]:


#cross-val score
score = cross_val_score(estimator=logit, X=X_train, y=y_train, cv=skf, scoring='roc_auc').mean()
print('ROC AUC score: ', round(score, 4))


# In[ ]:


def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# In[ ]:


logit.fit(X_train, y_train)
y_pred = logit.predict_proba(X_test)


# In[ ]:


y_pred[:, 1]


# In[ ]:


write_to_submission_file(y_pred[:, 1], 'submission.csv')

