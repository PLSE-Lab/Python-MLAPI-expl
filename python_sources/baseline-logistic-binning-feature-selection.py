#!/usr/bin/env python
# coding: utf-8

# This kernel shows some experiments I have done.
# 
# 1. basedline model logistic regression (result: overfitting)
# 2. feature binning (result: overfitting, but slightly improve the result)
# 3. feature selection (result: better, but still overfitting)

# ### 1. import packages and data

# In[27]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer

get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# ### 2. Baseline model

# In[29]:


train_x = train.drop(['id', 'target'], axis = 1)
train_y = train['target']
test_x = test.drop(["id"], axis = 1)


# In[30]:


def baseline_model(train_x, train_y, run_num = 10, fold = 5):
    train_result, test_result = [], []
    for i in range(run_num):
        # result list
        train_fold, test_fold = [], []
        # split dataset
        skf = StratifiedKFold(n_splits = fold, shuffle = True)
        fold_num = 1
        for train_index, valid_index in skf.split(train_x, train_y):
            # dataset
            X_train, X_valid = train_x.iloc[train_index], train_x.iloc[valid_index]
            y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]
            # model
            reg = LogisticRegression(solver = "liblinear", penalty = "l2")
            reg.fit(X_train, y_train)
            y_train_pred = reg.predict(X_train)
            y_valid_pred = reg.predict(X_valid)
            # result AUC
            train_auc = roc_auc_score(y_train, y_train_pred)
            test_auc = roc_auc_score(y_valid, y_valid_pred)
            if i == 1:
                print("TRAIN Fold {0}, AUC score: {1}".format(fold_num, round(train_auc, 4)))
                print("TEST Fold {0}, AUC score: {1}".format(fold_num, round(test_auc, 4)))
            fold_num += 1
            train_fold.append(train_auc)
            test_fold.append(test_auc)
        train_result.append(train_fold)
        test_result.append(test_fold)
    return train_result, test_result


# In[31]:


train_result, test_result = baseline_model(train_x = train_x, train_y = train_y, run_num = 10, fold = 5)


# In[32]:


def model_result(train_result, test_result):
    base_test_re = pd.DataFrame(test_result).T
    base_test_re.index = ['fold {0}'.format(i) for i in range(5)]
    base_test_re.columns = ['run {0}'.format(i) for i in range(10)]
    base_train_re = pd.DataFrame(train_result).T
    base_train_re.index = ['fold {0}'.format(i) for i in range(5)]
    base_train_re.columns = ['run {0}'.format(i) for i in range(10)]
    return base_train_re, base_test_re
base_train_re, base_test_re = model_result(train_result, test_result)


# In[33]:


base_train_re


# In[34]:


base_test_re.round(3)


# **Conclusion:** Overfitting..., I also tried to change logistic regression parameters, but it doesn't improve the score and cannot handle overfitting problem.

# ### 3. feature bining
# ```
# if x < quantile(0.1):
#      x = 0
# if quantile(0.1) < x < quantile(0.9)
#      x = 1-9
# if x > quantile(0.9):
#      x = 10
# ```

# In[35]:


def binning(data, feature, n_bins):
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit(data[feature].values)
    Xt = est.transform(data[feature].values)
    data[feature] = pd.DataFrame(Xt)
    return data


# In[36]:


train_x_bin = binning(train_x, train_x.columns, n_bins = 15)
test_x_bin = binning(test_x, test_x.columns, n_bins = 15)


# In[37]:


train_result_bin, test_result_bin = baseline_model(train_x_bin, train_y, run_num = 10, fold = 5)


# In[38]:


base_train_re_bin, base_test_re_bin = model_result(train_result, test_result)


# In[39]:


base_train_re_bin


# In[40]:


base_test_re_bin.round(3)


# **Conclusion:** I got better results but still overfitting...

# ### 3. feature selection by statistics test

# In[41]:


sig_features = []
for each_feature in train.columns[2:]:
    X = train[each_feature]
    X = sm.add_constant(X)
    y = train.iloc[:,1]
    model = sm.OLS(y, X)
    result = model.fit()
    pvalue = result.pvalues[1]
    # using 90% significance level
    if pvalue <= 0.1:
        print("Feature {0}, p value is {1}".format(each_feature, round(pvalue, 3)))
        sig_features.append(each_feature)


# In[42]:


train_x = train.drop(['id', 'target'], axis = 1)
train_y = train['target']


# In[43]:


train_select_x = train_x[sig_features]
train_select_bin_x = train_x_bin[sig_features]


# In[44]:


train_result_select, test_result_select = baseline_model(train_select_x, train_y, run_num = 10, fold = 5)
base_train_re_select, base_test_re_select = model_result(train_result_select, test_result_select)


# In[45]:


base_train_re_select


# In[46]:


base_test_re_select


# In[47]:


train_result_bin_select, test_result_bin_elect = baseline_model(train_select_bin_x, train_y, run_num = 10, fold = 5)
base_train_re_bin_select, base_test_re_bin_select = model_result(train_result_bin_select, test_result_bin_elect)


# In[48]:


base_train_re_bin_select


# In[49]:


base_test_re_bin_select


# ### 4. submission

# In[55]:


train_select_bin_x = train_x_bin[sig_features]
test_select_bin_x = test_x_bin[sig_features]


# In[58]:


# split dataset
skf = StratifiedKFold(n_splits = 5, shuffle = True)
fold_num = 1
y_test = np.zeros(len(test_select_bin_x))
for train_index, valid_index in skf.split(train_select_bin_x, train_y):
    # dataset
    X_train, X_valid = train_select_bin_x.iloc[train_index], train_select_bin_x.iloc[valid_index]
    y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]
    # model
    reg = LogisticRegression(solver = "liblinear", penalty = "l2")
    reg.fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    y_valid_pred = reg.predict(X_valid)
    # result AUC
    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_valid, y_valid_pred)
    fold_num += 1
    # predict test set
    y_test_fold = reg.predict_proba(test_select_bin_x)[:, 1]
    y_test += y_test_fold
y_test = y_test/5


# In[62]:


sub = pd.read_csv("../input/sample_submission.csv")
sub['target'] = y_test
sub.to_csv("submission_logit.csv", index = False)

