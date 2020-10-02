#!/usr/bin/env python
# coding: utf-8

# ***Please upvote this kernel if you like it!***
# 
# This is my initial experiment, so very simple process is performed:
# 
# 1. Check the correlation matrix
# 2. Select features that exceed a threshold value
# 3. Build Logistic Regression with selected features
# 
# Used this model : **https://www.kaggle.com/nadare/simple-logistic-regression-with-l1-penalty**

# In[ ]:


import numpy as np; np.random.random(42)
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# In[ ]:


plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['font.size'] = 12


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submit = pd.read_csv('../input/sample_submission.csv')
print(train.shape, test.shape, submit.shape)


# In[ ]:


print( train.target.value_counts() / train.shape[0] * 100 )


# In[ ]:


train.head()


# In[ ]:


feature_names = train.columns[2:]


# ## 1. Check the correlation matrix

# In[ ]:


plt.figure(figsize=(16,16))
train_corr = train.iloc[:, 1:].corr()
sns.heatmap(train_corr, cmap="Blues", square=True, vmax=1, vmin=-1, center=0)
plt.show()


# You can't capture any insight except that many features have a weak correlation.
# 
# So, focus on the correlation coefficient of features with target.

# In[ ]:


train_corr.iloc[[0],:]


# In[ ]:


sns.boxplot(train_corr.iloc[0, 1:])
plt.show()


# ## 2. Select features that exceed a threshold value

# get features that have over 0.1 or under -0.1 correlation coefficient

# In[ ]:


corr_index = (train_corr.iloc[0, 1:].values > 0.1) + (train_corr.iloc[0, 1:].values < -0.1)
features_selected = feature_names[corr_index].tolist()
print(len(features_selected))


# In[ ]:


train_selected = train.loc[:, ["target"]+features_selected]
test_selected = test.loc[:, features_selected]
train_selected.head()


# In[ ]:


sns.distplot(train.iloc[0,1:].values)
plt.show()


# In[ ]:


target = train_selected["target"].values
train_selected.drop(["target"], axis=1, inplace=True)


# ## 3. Build Logistic Regression with selected features

# In[ ]:


N_FOLDS = 10
features = train_selected.columns.tolist()


# In[ ]:


folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof = np.zeros(len(train_selected))
sub = np.zeros(len(test_selected))
score = [0 for _ in range(folds.n_splits)]

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_selected.values, target)):
    X_train, y_train = train_selected.iloc[trn_idx][features], target[trn_idx]
    X_val, y_val = train_selected.iloc[val_idx][features], target[val_idx]
    X_test = test_selected.values
    clf = LogisticRegression(penalty="l1", C=0.1, solver="liblinear", random_state=42)
    clf.fit(X_train, y_train)
    oof[val_idx] = clf.predict_proba(X_val)[:, 1]
    sub += clf.predict_proba(X_test)[:, 1] / folds.n_splits
    score[fold_] = roc_auc_score(target[val_idx], oof[val_idx])
    print("Fold {}: {}".format(fold_+1, round(score[fold_],5)))

print("CV score(auc): {:<8.5f}, (std: {:<8.5f})".format(roc_auc_score(target, oof), np.std(score)))


# Check the distribution of the prediction

# In[ ]:


sns.boxplot(sub)
plt.show()


# In[ ]:


submit = pd.read_csv('../input/sample_submission.csv')
submit["target"] = sub
submit.to_csv("submission.csv", index=False)
submit.head(20)


# **Thank you for reading!**

# In[ ]:




