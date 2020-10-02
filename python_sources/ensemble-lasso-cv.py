#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline

from collections import OrderedDict
import os
print(os.listdir("../input"))


# In[3]:


train_df = pd.read_csv('../input/train.csv', index_col='id')
test_df = pd.read_csv('../input/test.csv', index_col='id')


# In[4]:


train_df.head()


# In[5]:


test_df.head()


# In[6]:


print("number of training samples: ", train_df.shape[0])
print("number of testing samples: ", test_df.shape[0])


# In[7]:


print("count of target values: ")
train_df.target.value_counts()


# In[8]:


for ix in range(2,12): # (2, 302) for all features
    fig, ax = plt.subplots(figsize=(6,6))
    sns.distplot(train_df.iloc[:,ix], ax=ax, fit=norm, rug=True, label="train");
    sns.distplot(test_df.iloc[:,ix], ax=ax, fit=norm, rug=True, label="test");
    plt.legend()
    plt.show()


# In[9]:


def bootstrap(data, target, n=5):
    """
    bootstrap samples for use testing model generalization
    """
    samples = []
    for i in range(n):
        ix = range(len(data))
        random_ixs = np.random.choice(ix, len(data), replace=True)
        new_data, new_target = data[random_ixs, :], target[random_ixs]
        samples.append((new_data, new_target))
    return samples


# In[10]:


features = [col for col in train_df.columns if col.isdigit()]
target = 'target'


# In[41]:


mean_train_auc_roc = []
mean_val_auc_roc = []

alpha_values = np.logspace(-3, -1, 20)
for a in alpha_values:
    # bootstrap_samples = bootstrap(X_train, y_train, 1000)
    # for sample in bootstrap_samples:
    #     X, y = sample

    X, y = train_df[features].values, train_df[target].values
    skf = StratifiedKFold(n_splits=5)

    # train_auc_roc = OrderedDict((label, []) for label, _ in clfs)
    # val_auc_roc = OrderedDict((label, []) for label, _ in clfs)
    train_auc_roc = []
    val_auc_roc = []

    for train_ix, val_ix in skf.split(X, y):
        X_train, y_train, X_val, y_val = X[train_ix], y[train_ix], X[val_ix], y[val_ix]
        lr = Lasso(alpha=a, normalize=False, max_iter=1000, precompute=True)
        lr.fit(X_train, y_train)
        y_train_pred = lr.predict(X_train)
        y_val_pred = lr.predict(X_val)
        train_auc_roc.append(roc_auc_score(y_train, y_train_pred))
        val_auc_roc.append(roc_auc_score(y_val, y_val_pred))
    
    mean_train_auc_roc.append(np.mean(train_auc_roc))
    mean_val_auc_roc.append(np.mean(val_auc_roc))


# In[42]:


max_alpha_ix = np.argmax(np.vstack((mean_train_auc_roc, mean_val_auc_roc)).mean(axis=0))
max_mean_cv_score = np.max(np.vstack((mean_train_auc_roc, mean_val_auc_roc)).mean(axis=0))
alpha = alpha_values[max_alpha_ix]
print("alpha: ", alpha, "max_mean, cv_score: ", max_mean_cv_score)


# In[26]:


plt.figure(1)
plt.plot(alpha_values, mean_train_auc_roc, label='train')
plt.plot(alpha_values, mean_val_auc_roc, label='validation')
plt.axvline(x=alpha, lw=.2, color='red', label='chosen_alpha')
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('auc_roc')
plt.title('Area under ROC curve for Lasso regression')
plt.legend(loc='best')
plt.show()


# In[65]:


mean_train_auc_roc = []
mean_val_auc_roc = []

alpha_values = np.logspace(-3, -1, 20)
for a in alpha_values:
    # bootstrap_samples = bootstrap(X_train, y_train, 1000)
    # for sample in bootstrap_samples:
    #     X, y = sample

    X, y = train_df[features].values, train_df[target].values
    skf = StratifiedKFold(n_splits=5)

    # train_auc_roc = OrderedDict((label, []) for label, _ in clfs)
    # val_auc_roc = OrderedDict((label, []) for label, _ in clfs)
    train_auc_roc = []
    val_auc_roc = []

    for train_ix, val_ix in skf.split(X, y):
        X_train, y_train, X_val, y_val = X[train_ix], y[train_ix], X[val_ix], y[val_ix]
        lr = ElasticNet(alpha=a, l1_ratio=0.75, normalize=False, max_iter=1000, precompute=True)
        lr.fit(X_train, y_train)
        y_train_pred = lr.predict(X_train)
        y_val_pred = lr.predict(X_val)
        train_auc_roc.append(roc_auc_score(y_train, y_train_pred))
        val_auc_roc.append(roc_auc_score(y_val, y_val_pred))
    
    mean_train_auc_roc.append(np.mean(train_auc_roc))
    mean_val_auc_roc.append(np.mean(val_auc_roc))


# In[66]:


max_alpha_ix = np.argmax(np.vstack((mean_train_auc_roc, mean_val_auc_roc)).mean(axis=0))
max_mean_cv_score = np.max(np.vstack((mean_train_auc_roc, mean_val_auc_roc)).mean(axis=0))
alpha = alpha_values[max_alpha_ix]
print("alpha: ", alpha, "max_mean, cv_score: ", max_mean_cv_score)


# In[67]:


plt.figure(1)
plt.plot(alpha_values, mean_train_auc_roc, label='train')
plt.plot(alpha_values, mean_val_auc_roc, label='validation')
plt.axvline(x=alpha, lw=.2, color='red', label='chosen_alpha')
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('auc_roc')
plt.title('Area under ROC curve for Elastic Net')
plt.legend(loc='best')
plt.show()


# In[52]:


lr = ElasticNet(alpha=alpha, l1_ratio=0.75, normalize=False, max_iter=1000, precompute=True)
lr.fit(X_train, y_train)
test_predictions = lr.predict(test_df.values)


# In[ ]:


submission = pd.DataFrame({
    'id': np.arange(250,20000),
    'target': test_predictions
})
submission.to_csv("submission.csv", index=False)


# In[ ]:




