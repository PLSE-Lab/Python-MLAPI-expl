#!/usr/bin/env python
# coding: utf-8

# # Import data

# In[279]:


import pandas as pd

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()


# In[280]:


print(set(train["wheezy-copper-turtle-magic"]))


# # The strategy
# 
# Following many different source ([Chris Deotte's post](https://www.kaggle.com/c/instant-gratification/discussion/92930#latest-534847), [Chris Deotte's kernel](https://www.kaggle.com/cdeotte/support-vector-machine-0-925)), a distinct model is built for each distinct value of variable $\mathrm{wheezy-copper-turtle-magic}$. Let's start with model 0.

# ## Model 0

# In[336]:


train_1 = train.loc[train["wheezy-copper-turtle-magic"]==1,].copy()
test_1 = test.loc[test["wheezy-copper-turtle-magic"]==1,].copy()
train_1.shape


# In[337]:


test_1.shape


# In[338]:


train_1.head()


# The number of observations is comparable to the number of columns: this require some classifiers like Random Forest or SVM. Preparing data:

# In[339]:


Y = train_1["target"]
X = train_1.drop(["id", "target", "wheezy-copper-turtle-magic"], axis=1)
X.shape


# In[340]:


import numpy as np
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(n_estimators=10)
scores_rf = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_rf), "+/-", np.std(scores_rf))


# In[341]:


import seaborn as sns

sns.boxplot(x=scores_rf)


# In[342]:


model = RandomForestClassifier(n_estimators=100)
scores_rf = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_rf), "+/-", np.std(scores_rf))


# In[343]:


model = RandomForestClassifier(n_estimators=200)
scores_rf = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_rf), "+/-", np.std(scores_rf))


# In[344]:


model = RandomForestClassifier(n_estimators=400)
scores_rf = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_rf), "+/-", np.std(scores_rf))


# In[345]:


model = RandomForestClassifier(n_estimators=600)
scores_rf = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_rf), "+/-", np.std(scores_rf))


# Trying with SVM:

# In[346]:


from sklearn import svm

model = svm.SVC()
scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_svm), "+/-", np.std(scores_svm))


# In[347]:


model


# In[348]:


model = svm.SVC(kernel='linear')
scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_svm), "+/-", np.std(scores_svm))


# In[349]:


model = svm.SVC(kernel='poly')
scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_svm), "+/-", np.std(scores_svm))


# In[350]:


model = svm.SVC(kernel='poly', degree=2)
scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_svm), "+/-", np.std(scores_svm))


# In[351]:


model = svm.SVC(kernel='poly', degree=4)
scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_svm), "+/-", np.std(scores_svm))


# In[352]:


model = svm.SVC(kernel='poly', degree=5)
scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_svm), "+/-", np.std(scores_svm))


# Thus the most promising approach is to use a polynomial of degree 4 as kernel.

# In[353]:


model = svm.SVC(kernel='poly', degree=4)
scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

sns.boxplot(x=scores_svm)


# ## Adding Features

# In[354]:


columns = train.columns[1:-1]

first_name = [i.split("-")[0] for i in columns]
print(set(first_name))
print(len(first_name))
print(len(set(first_name)))
for first in first_name:
    filter_col = [col for col in train_1 if col.startswith(first)]
    test_1.loc[:, first+"-mean"] = test_1.loc[:, filter_col].mean(axis=1)
    train_1.loc[:, first+"-mean"] = train_1.loc[:, filter_col].mean(axis=1)
    test_1.loc[:, first+"-std"] = test_1.loc[:, filter_col].std(axis=1)
    train_1.loc[:, first+"-std"] = train_1.loc[:, filter_col].std(axis=1)


# In[355]:


second_name = [i.split("-")[1] for i in columns]
print(set(second_name))
print(len(second_name))
print(len(set(second_name)))
for second in second_name:
    filter_col = [col for col in columns if second==col.split("-")[1]]
    test_1[second+"-mean"] = test_1.loc[:, filter_col].mean(axis=1)
    train_1[second+"-mean"] = train_1.loc[:, filter_col].mean(axis=1)
    test_1[second+"-std"] = test_1.loc[:, filter_col].std(axis=1)
    train_1[second+"-std"] = train_1.loc[:, filter_col].std(axis=1)


# In[356]:


third_name = [i.split("-")[2] for i in columns]
print(set(third_name))
print(len(third_name))
print(len(set(third_name)))
for third in third_name:
    filter_col = [col for col in columns if third==col.split("-")[1]]
    test_1[third+"-mean"] = test_1.loc[:, filter_col].mean(axis=1)
    train_1[third+"-mean"] = train_1.loc[:, filter_col].mean(axis=1)
    test_1[third+"-std"] = test_1.loc[:, filter_col].std(axis=1)
    train_1[third+"-std"] = train_1.loc[:, filter_col].std(axis=1)


# In[357]:


fourth_name = [i.split("-")[3] for i in columns]
print(set(fourth_name))
print(len(fourth_name))
print(len(set(fourth_name)))
for fourth in fourth_name:
    filter_col = [col for col in columns if fourth==col.split("-")[1]]
    test_1[fourth+"-mean"] = test_1.loc[:, filter_col].mean(axis=1)
    train_1[fourth+"-mean"] = train_1.loc[:, filter_col].mean(axis=1)
    test_1[fourth+"-std"] = test_1.loc[:, filter_col].std(axis=1)
    train_1[fourth+"-std"] = train_1.loc[:, filter_col].std(axis=1)


# Some keywords appear just once, so they can be dropped:

# In[358]:


for col in train_1.columns:
    if (train_1[col].isnull().sum()>0):
        train_1.drop([col], axis=1, inplace=True)
        test_1.drop([col], axis=1, inplace=True)

train_1.shape


# In[359]:


test_1.shape


# ## Retrained a SVM model

# In[360]:


Y = train_1["target"]
X = train_1.drop(["id", "target", "wheezy-copper-turtle-magic"], axis=1)
X.shape

model = svm.SVC(kernel='poly', degree=4)
scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_svm), "+/-", np.std(scores_svm))


# ## Retrained a Random Forest model

# In[361]:


model = RandomForestClassifier(n_estimators=400)
scores_rf = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_rf), "+/-", np.std(scores_rf))


# In[362]:


model = RandomForestClassifier(n_estimators=400)
model.fit(X,Y)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))


# ## Train Xgboost

# In[363]:


from xgboost import XGBClassifier

model = XGBClassifier(njobs=-1)
scores_xgb = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_xgb), "+/-", np.std(scores_xgb))


# In[364]:


model


# In[365]:


model = XGBClassifier(learning_rate=0.01, n_estimators=1000)
scores_xgb = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_xgb), "+/-", np.std(scores_xgb))


# # Traing Lightgbm

# In[366]:


from lightgbm import LGBMClassifier

model = LGBMClassifier(njobs=-1)
scores_gbm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_gbm), "+/-", np.std(scores_gbm))


# In[367]:


model


# ## Adversarial validation
# 
# Can I [train a classifier to distinguish between train/test](http://fastml.com/adversarial-validation-part-one/)? If that is the case, the relevant features are thos that are "different" between train and test. A value of the AUC of 0.5 would lead to the conclusion of no significant difference between train/test, a value close to 1 as train and test sets radically different.

# In[368]:


X2 = test_1.drop(["id", "wheezy-copper-turtle-magic"], axis=1)
X2.shape


# In[369]:


X.shape


# In[370]:


adv = pd.concat((X, X2), axis=0)
adv.head()


# In[371]:


adv.shape


# In[372]:


label = ["0"]*510+["1"]*250

model = svm.SVC(kernel='poly', degree=4)
scores_svm = cross_val_score(model, adv, label, cv=10, n_jobs=-1, scoring='roc_auc')
print(scores_svm)
print(np.mean(scores_svm), "+/-", np.std(scores_svm))


# In[373]:


model = RandomForestClassifier(n_estimators=400)
scores_rf = cross_val_score(model, adv, label, cv=10, n_jobs=-1, scoring='roc_auc')
print(np.mean(scores_rf), "+/-", np.std(scores_rf))


# In[378]:


model = RandomForestClassifier(n_estimators=400).fit(adv, label)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(adv.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, adv.columns[indices[f]], importances[indices[f]]))


# In[379]:


sns.distplot(adv.loc[:, "crappy-carmine-eagle-entropy"][0:510])
sns.distplot(adv.loc[:, "crappy-carmine-eagle-entropy"][510:760])


# The difference does not seems to be significant.
