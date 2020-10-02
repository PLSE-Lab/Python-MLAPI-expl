#!/usr/bin/env python
# coding: utf-8

# # Building a model
# 
# This is a fork from [nvnn's kernel](https://www.kaggle.com/nvnnghia/svm-knn-0-943) supplemented with [Chris Deotte's idea for feature selection](https://www.kaggle.com/c/instant-gratification/discussion/92930):

# In[ ]:


import pandas as pd

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()


# In[124]:


import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


submission=test[["id"]].copy()
submission["target"] = -1.0
performance_report = pd.DataFrame({
    "segment": range(512),
    "mean": 0,
    "std": 0
})

for i in range(10,11):
    Y = train.loc[train["wheezy-copper-turtle-magic"]==i, "target"]
    X = train.loc[train["wheezy-copper-turtle-magic"]==i, :].drop(["id", "target", "wheezy-copper-turtle-magic"], axis=1)
    X_te = test.loc[test["wheezy-copper-turtle-magic"]==i, :].drop(["id", "wheezy-copper-turtle-magic"], axis=1)
    selected_columns = X.columns[(X.std(axis=0)>2).values]
    X = X.loc[:, selected_columns]
    X_te = X_te[selected_columns]
    
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    scores_rf = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
    print(np.mean(scores_rf), "+/-", np.std(scores_rf))
    
    model = svm.SVC(kernel='poly', degree=4, probability=True, gamma='auto', random_state=1)
    scores_svm = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
    print(np.mean(scores_svm), "+/-", np.std(scores_svm))
    
    model = LogisticRegression(solver='liblinear', penalty="l1", random_state=1)
    scores_lr = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
    print(np.mean(scores_lr), "+/-", np.std(scores_lr))
    
    model = KNeighborsClassifier(n_neighbors=12)
    scores_knn = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
    print(np.mean(scores_knn), "+/-", np.std(scores_knn))
    
    print(np.mean(scores_lr+scores_knn+scores_svm+scores_rf)/4)
#    performance_report.loc[i, "mean"] = np.mean(scores_rf)
#    performance_report.loc[i, "std"] = np.std(scores_rf)
#    model = RandomForestClassifier(n_estimators=100, n_jobs=-1).fit(X, Y)
#    submission.loc[test["wheezy-copper-turtle-magic"]==i, "target"] =  model.predict_proba(X_te)[:,1]


# In[68]:


performance_report.to_csv("performance.csv", index=False)
submission.to_csv("submission.csv", index=False)


# In[71]:


print(model)


# In[125]:


Y = train.loc[train["wheezy-copper-turtle-magic"]==i, "target"]
X = train.loc[train["wheezy-copper-turtle-magic"]==i, :].drop(["id", "target", "wheezy-copper-turtle-magic"], axis=1)
X_te = test.loc[test["wheezy-copper-turtle-magic"]==i, :].drop(["id", "wheezy-copper-turtle-magic"], axis=1)


# In[88]:


from sklearn import mixture

dpgmm = mixture.BayesianGaussianMixture(n_components=X.shape[1], covariance_type='full').fit(X)
print(len(set(dpgmm.predict(X))))


# In[89]:


print(dpgmm.weights_)


# In[94]:


import seaborn as sns

sns.distplot(dpgmm.weights_)


# In[98]:


print(np.sum(dpgmm.weights_ > np.quantile(dpgmm.weights_, 0.95)))


# In[150]:


dpgmm = mixture.BayesianGaussianMixture(n_components=X.shape[1], covariance_type='full', random_state=2019).fit(X)
dpgmm = mixture.BayesianGaussianMixture(n_components=np.sum(dpgmm.weights_ > np.quantile(dpgmm.weights_, 0.95)), 
                                        covariance_type='full', random_state=2019).fit(X)


# In[160]:


clusters = pd.DataFrame({
    "cluster_name": dpgmm.predict(X),
    "Y": Y
})
dic = clusters.groupby("cluster_name").mean()


# In[164]:


dic.iloc[1,0]


# # Using nearest neighbours

# In[153]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
scores_knn = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_knn), "+/-", np.std(scores_knn))


# In[154]:


model = KNeighborsClassifier(n_neighbors=4)
scores_knn = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_knn), "+/-", np.std(scores_knn))


# In[155]:


model = KNeighborsClassifier(n_neighbors=10)
scores_knn = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_knn), "+/-", np.std(scores_knn))


# In[156]:


model = KNeighborsClassifier(n_neighbors=15)
scores_knn = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_knn), "+/-", np.std(scores_knn))


# In[157]:


model = KNeighborsClassifier(n_neighbors=12)
scores_knn = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_knn), "+/-", np.std(scores_knn))


# In[158]:


model = KNeighborsClassifier(n_neighbors=8)
scores_knn = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_knn), "+/-", np.std(scores_knn))


# # Classifiers with new column
# 
# Now, comparing all the classifiers with the new column (and trying to fine tune the amount of components to be considered in the clustering):

# In[182]:


Y = train.loc[train["wheezy-copper-turtle-magic"]==i, "target"]
X = train.loc[train["wheezy-copper-turtle-magic"]==i, :].drop(["id", "target", "wheezy-copper-turtle-magic"], axis=1)
X_te = test.loc[test["wheezy-copper-turtle-magic"]==i, :].drop(["id", "wheezy-copper-turtle-magic"], axis=1)

selected_columns = X.columns[(X.std(axis=0)>2).values]
X = X.loc[:, selected_columns]
X_te = X_te[selected_columns]

dpgmm = mixture.BayesianGaussianMixture(n_components=X.shape[1], covariance_type='full', random_state=2019).fit(X)
dpgmm = mixture.BayesianGaussianMixture(n_components=np.sum(dpgmm.weights_ > np.quantile(dpgmm.weights_, 0.95)), 
                                        covariance_type='full', random_state=2019).fit(X)
dic = pd.DataFrame({"cluster_name": dpgmm.predict(X), "Y": Y}).groupby("cluster_name").mean()

column_train = [dic.iloc[i, 0] for i in dpgmm.predict(X)]
column_test = [dic.iloc[i, 0] for i in dpgmm.predict(X_te)]

X["train_column"] = column_train
X_te["train_column"] = column_test

model = RandomForestClassifier(n_estimators=100, random_state=1)
scores_rf = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_rf), "+/-", np.std(scores_rf))

model = svm.SVC(kernel='poly', degree=4, probability=True, gamma='auto', random_state=1)
scores_svm = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_svm), "+/-", np.std(scores_svm))
    
model = LogisticRegression(solver='liblinear', penalty="l1", random_state=1)
scores_lr = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_lr), "+/-", np.std(scores_lr))
    
model = KNeighborsClassifier(n_neighbors=12)
scores_knn = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_knn), "+/-", np.std(scores_knn))


# In[183]:


Y = train.loc[train["wheezy-copper-turtle-magic"]==i, "target"]
X = train.loc[train["wheezy-copper-turtle-magic"]==i, :].drop(["id", "target", "wheezy-copper-turtle-magic"], axis=1)
X_te = test.loc[test["wheezy-copper-turtle-magic"]==i, :].drop(["id", "wheezy-copper-turtle-magic"], axis=1)

dpgmm = mixture.BayesianGaussianMixture(n_components=X.shape[1], covariance_type='full', random_state=2019).fit(X)
dpgmm = mixture.BayesianGaussianMixture(n_components=np.sum(dpgmm.weights_ > np.quantile(dpgmm.weights_, 0.95)), 
                                        covariance_type='full', random_state=2019).fit(X)
dic = pd.DataFrame({"cluster_name": dpgmm.predict(X), "Y": Y}).groupby("cluster_name").mean()

column_train = [dic.iloc[i, 0] for i in dpgmm.predict(X)]
column_test = [dic.iloc[i, 0] for i in dpgmm.predict(X_te)]

X["train_column"] = column_train
X_te["train_column"] = column_test

model = RandomForestClassifier(n_estimators=100, random_state=1)
scores_rf = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_rf), "+/-", np.std(scores_rf))

model = svm.SVC(kernel='poly', degree=4, probability=True, gamma='auto', random_state=1)
scores_svm = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_svm), "+/-", np.std(scores_svm))
    
model = LogisticRegression(solver='liblinear',random_state=1)
scores_lr = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_lr), "+/-", np.std(scores_lr))
    
model = KNeighborsClassifier(n_neighbors=12)
scores_knn = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_knn), "+/-", np.std(scores_knn))


# In[188]:


Y = train.loc[train["wheezy-copper-turtle-magic"]==i, "target"]
X = train.loc[train["wheezy-copper-turtle-magic"]==i, :].drop(["id", "target", "wheezy-copper-turtle-magic"], axis=1)
X_te = test.loc[test["wheezy-copper-turtle-magic"]==i, :].drop(["id", "wheezy-copper-turtle-magic"], axis=1)

dpgmm = mixture.BayesianGaussianMixture(n_components=X.shape[1], covariance_type='full', random_state=2019).fit(X)
dpgmm = mixture.BayesianGaussianMixture(n_components=np.sum(dpgmm.weights_ > np.quantile(dpgmm.weights_, 0.90)), 
                                        covariance_type='full', random_state=2019).fit(X)
dic = pd.DataFrame({"cluster_name": dpgmm.predict(X), "Y": Y}).groupby("cluster_name").mean()

column_train = [dic.iloc[i, 0] for i in dpgmm.predict(X)]
column_test = [dic.iloc[i, 0] for i in dpgmm.predict(X_te)]

X["train_column"] = column_train
X_te["train_column"] = column_test

model = RandomForestClassifier(n_estimators=100, random_state=1)
scores_rf = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_rf), "+/-", np.std(scores_rf))

model = svm.SVC(kernel='poly', degree=4, probability=True, gamma='auto', random_state=1)
scores_svm = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_svm), "+/-", np.std(scores_svm))
    
model = LogisticRegression(solver='liblinear',random_state=1)
scores_lr = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_lr), "+/-", np.std(scores_lr))
    
model = KNeighborsClassifier(n_neighbors=12)
scores_knn = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_knn), "+/-", np.std(scores_knn))


# In[189]:


Y = train.loc[train["wheezy-copper-turtle-magic"]==i, "target"]
X = train.loc[train["wheezy-copper-turtle-magic"]==i, :].drop(["id", "target", "wheezy-copper-turtle-magic"], axis=1)
X_te = test.loc[test["wheezy-copper-turtle-magic"]==i, :].drop(["id", "wheezy-copper-turtle-magic"], axis=1)

dpgmm = mixture.BayesianGaussianMixture(n_components=X.shape[1], covariance_type='full', random_state=2019).fit(X)
dpgmm = mixture.BayesianGaussianMixture(n_components=np.sum(dpgmm.weights_ > np.quantile(dpgmm.weights_, 0.85)), 
                                        covariance_type='full', random_state=2019).fit(X)
dic = pd.DataFrame({"cluster_name": dpgmm.predict(X), "Y": Y}).groupby("cluster_name").mean()

column_train = [dic.iloc[i, 0] for i in dpgmm.predict(X)]
column_test = [dic.iloc[i, 0] for i in dpgmm.predict(X_te)]

X["train_column"] = column_train
X_te["train_column"] = column_test

model = RandomForestClassifier(n_estimators=100, random_state=1)
scores_rf = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_rf), "+/-", np.std(scores_rf))

model = svm.SVC(kernel='poly', degree=4, probability=True, gamma='auto', random_state=1)
scores_svm = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_svm), "+/-", np.std(scores_svm))
    
model = LogisticRegression(solver='liblinear',random_state=1)
scores_lr = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_lr), "+/-", np.std(scores_lr))
    
model = KNeighborsClassifier(n_neighbors=12)
scores_knn = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_knn), "+/-", np.std(scores_knn))


# Basically, the SVM seems slightly better with the feature. It should added to the others.

# # Gradient Boosting
# 
# Trying algorithms that uses gradient boosting to assess the performance (without fine-tuning):

# In[186]:


from xgboost import XGBClassifier

model = XGBClassifier()
scores_xgb = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_xgb), "+/-", np.std(scores_xgb))


# In[187]:


from lightgbm import LGBMClassifier

model = LGBMClassifier()
scores_gbm = cross_val_score(model, X, Y, cv=10, scoring='roc_auc')
print(np.mean(scores_gbm), "+/-", np.std(scores_gbm))


# In[ ]:




