#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/train.csv', index_col='id')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.target.value_counts()


# In[ ]:


for ix in range(2,12): # (2, 302) for all features
    fig, ax = plt.subplots(figsize=(6,6))
    sns.distplot(df.iloc[:,ix], ax=ax, fit=norm, rug=True);
    plt.show()


# In[ ]:


def bootstrap(data, target, n=5):
    samples = []
    for i in range(n):
        ix = range(len(data))
        random_ixs = np.random.choice(ix, len(X), replace=True)
        new_data, new_target = data[random_ixs, :], target[random_ixs]
        samples.append((new_data, new_target))
    return samples


# In[ ]:


X = df.iloc[:, 1:].values
y = df.target.values


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5)
bootstrap_samples = bootstrap(X_train, y_train, 1000)


# In[ ]:


n_estimator = 100


# https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py

# In[ ]:


list_y_pred_rt = []
list_y_pred_rf_lm = []
list_y_pred_grd_lm = []
list_y_pred_grd = []
list_y_pred_rf = []

for sample in bootstrap_samples:
    X_train, y_train = sample
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(
                                        X_train, y_train, test_size=0.5)
    
    rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
                          random_state=0)
    rt_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
    pipeline = make_pipeline(rt, rt_lm)
    rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
    rf_enc = OneHotEncoder(categories='auto')
    rf_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
    grd = GradientBoostingClassifier(n_estimators=n_estimator)
    grd_enc = OneHotEncoder(categories='auto')
    grd_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
    
    pipeline.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    rf_enc.fit(rf.apply(X_train))
    rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
    grd.fit(X_train, y_train)
    grd_enc.fit(grd.apply(X_train)[:, :, 0])
    grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

    y_pred_rt = pipeline.predict_proba(X_val)[:, 1]
    y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_val)))[:, 1]
    y_pred_grd_lm = grd_lm.predict_proba(
                        grd_enc.transform(grd.apply(X_val)[:, :, 0]))[:, 1]
    y_pred_grd = grd.predict_proba(X_val)[:, 1]
    y_pred_rf = rf.predict_proba(X_val)[:, 1]
    
    list_y_pred_rt.append(y_pred_rt)
    list_y_pred_rf_lm.append(y_pred_rf_lm)
    list_y_pred_grd_lm.append(y_pred_grd_lm)
    list_y_pred_grd.append(y_pred_grd)
    list_y_pred_rf.append(y_pred_rf)

y_pred_rt = np.array(list_y_pred_rt).mean(axis=0)
y_pred_rf_lm = np.array(list_y_pred_rf_lm).mean(axis=0)
y_pred_grd_lm = np.array(list_y_pred_grd_lm).mean(axis=0)
y_pred_grd = np.array(list_y_pred_grd).mean(axis=0)
y_pred_rf = np.array(list_y_pred_rf).mean(axis=0)

fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_val, y_pred_rt)
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_val, y_pred_rf_lm)
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_val, y_pred_grd_lm)
fpr_grd, tpr_grd, _ = roc_curve(y_val, y_pred_grd)
fpr_rf, tpr_rf, _ = roc_curve(y_val, y_pred_rf)


# In[ ]:


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[ ]:


test_df = pd.read_csv('../input/test.csv', index_col='id')


# In[ ]:


test_df.head()


# In[ ]:


X_test = test_df.values


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5)
bootstrap_samples = bootstrap(X_train, y_train, 1000)


# In[ ]:


list_y_pred_rf_lm = []

for sample in bootstrap_samples:
    X_train, y_train = sample
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(
                                        X_train, y_train, test_size=0.5)
    
    rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
    rf_enc = OneHotEncoder(categories='auto')
    rf_lm = LogisticRegression(solver='lbfgs', max_iter=1000)

    rf.fit(X_train, y_train)
    rf_enc.fit(rf.apply(X_train))
    rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

    y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]

    list_y_pred_rf_lm.append(y_pred_rf_lm)

y_pred_rf_lm = np.array(list_y_pred_rf_lm).mean(axis=0)


# In[ ]:


submission = pd.DataFrame({
    'id': np.arange(250,20000),
    'target': y_pred_rf_lm
})
submission.to_csv("submission.csv", index=False)


# In[ ]:




