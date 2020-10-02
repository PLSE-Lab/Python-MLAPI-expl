#!/usr/bin/env python
# coding: utf-8

# **Introduce WineQuality Machine learning Using Ensemble methods - RandomForest, GradientBoosting, XGBoost**
# 
# by. YHJ
# 
#  
# Basic Classification
# 1. Logistic Regression
# 2. Decision Tree
# 3. SVM (use GridSearchCV)
# 
# Ensemble methods
# 4. Random Forest (use RandomizedSearchCV)
# 5. GradientBoosting (use RandomizedSearchCV)
# 6. XGBoost (use RandomizedSearchCV)
#  

# In[ ]:


import numpy as np
import pandas as pd
from time import time
import scipy.stats as st

import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# data Read
data = pd.read_csv('../input/winequality-red.csv')


# In[ ]:


# data Check
data.head(10)


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


# Correlation Analysis
data.corr()


# In[ ]:


data.corr()['quality'].sort_values(ascending=False)


# In[ ]:


# Seaborn Heatmap, cmap param = Blues, Greys, OrRd, RdBu_r, Reds
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), cmap='RdBu_r', annot=True, linewidths=0.5)


# In[ ]:


# volatile acidity, citric acid, sulphates, alcohol
# fixed acidity, chlorides, total sulfur dioxide, density
# Use Boxplot
fig, axs = plt.subplots(2, 4, figsize = (20,10)) 
ax1 = plt.subplot2grid((5,15), (0,0), rowspan=2, colspan=3) 
ax2 = plt.subplot2grid((5,15), (0,4), rowspan=2, colspan=3)
ax3 = plt.subplot2grid((5,15), (0,8), rowspan=2, colspan=3)
ax4 = plt.subplot2grid((5,15), (0,12), rowspan=2, colspan=3)

ax5 = plt.subplot2grid((5,15), (3,0), rowspan=2, colspan=3) 
ax6 = plt.subplot2grid((5,15), (3,4), rowspan=2, colspan=3)
ax7 = plt.subplot2grid((5,15), (3,8), rowspan=2, colspan=3)
ax8 = plt.subplot2grid((5,15), (3,12), rowspan=3, colspan=3)

sns.boxplot(x='quality',y='volatile acidity', data = data, ax=ax1)
sns.boxplot(x='quality',y='citric acid', data = data, ax=ax2)
sns.boxplot(x='quality',y='sulphates', data = data, ax=ax3)
sns.boxplot(x='quality',y='alcohol', data = data, ax=ax4)

sns.boxplot(x='quality',y='fixed acidity', data = data, ax=ax5)
sns.boxplot(x='quality',y='chlorides', data = data, ax=ax6)
sns.boxplot(x='quality',y='total sulfur dioxide', data = data, ax=ax7)
sns.boxplot(x='quality',y='density', data = data, ax=ax8)


# In[ ]:


# See the Quality value
sns.countplot(x='quality', data=data)


# In[ ]:


# Dividing Quality as bad(0), good(1), Excellent(2)
# Bad quality is 3~4 / Good quality is 5~6 / Excellent quality is 7~8
data['quality'] = pd.cut(data['quality'], bins = [1,4.5,6.5,10], labels = [0,1,2])
sns.countplot(x='quality', data=data)


# In[ ]:


X = data.iloc[:,:-1]
y = data['quality']


# In[ ]:


# Sscale standization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[ ]:


# PCA - Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X)


# In[ ]:


# PCA Components - Barplot
plt.figure(figsize=(6,6))
sns.barplot(x=list(range(len(pca.explained_variance_))), y=pca.explained_variance_, palette="Blues_d")
plt.ylabel('Explained variance Value')
plt.xlabel('Principal components')
plt.grid(True)
plt.tight_layout()


# In[ ]:


# PCA Components Ratio - plot
plt.figure(figsize=(6,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.grid(True)
plt.tight_layout()


# In[ ]:


# Components 0 ~ 8 is explain the data 95% more
pca_comp = PCA(n_components = 8)
X = pca_comp.fit_transform(X)


# Look PCA ratio plot. We Use Components 0 ~ 8
# 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)


# In[ ]:


# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)


# In[ ]:


conf_matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)*100
print(conf_matrix)
print('\nAccuracy : %0.2f %%\n' % acc)
print(classification_report(y_test, y_pred))


# **1. Logistic Regression**
# * Accuracy = 83.5 %

# 

# In[ ]:


# 2. Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth = 5)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)


# In[ ]:


conf_matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)*100
print(conf_matrix)
print('\nAccuracy : %0.2f %%\n' % acc)
print(classification_report(y_test, y_pred))


# **2. Decision Tree**
# * Accuracy = 82.25 %

# 

# In[ ]:


# GridSearchCV, RandomizedSearchCV Report Function -> by. scikit-learn.org "Comparing randomized search and grid search for hyperparameter estimation"
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[ ]:


# 3. SVC / GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# SVM - Classifier
svc = SVC()

# Search Bast param
param_dist = {
        'C'      : [0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 5, 10, 100], 
        'gamma'  : [0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 5, 10, 100], 
}

# CAUTION! GridSearchCV is takes a lot of resources and time.
# if you want faster result, using param 'n_jobs = -1'
# GSCV_svc = GridSearchCV(svc, param_dist, scoring='accuracy', cv=10, n_jobs=-1)
GSCV_svc = GridSearchCV(svc, param_dist, scoring='accuracy', cv=10)


# In[ ]:


start = time()
GSCV_svc.fit(X_train, y_train)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(GSCV_svc.cv_results_['params'])))
report(GSCV_svc.cv_results_)


# In[ ]:


y_pred = GSCV_svc.predict(X_test)


# In[ ]:


conf_matrix = metrics.confusion_matrix(y_test,y_pred)
acc = metrics.accuracy_score(y_test, y_pred)*100
print(conf_matrix)
print('\nAccuracy : %0.2f %%\n' % acc)
print(classification_report(y_test, y_pred))


# **3. SVClassifier - GridSearchCV**
# * Accuracy = 87.5 %

# 

# In[ ]:


# 4. Random Forest / RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
rf = RandomForestClassifier()

param_dist = {
        'n_estimators': st.randint(10, 100),
        'max_depth' : [3, None],
        'max_features' : st.randint(1, 8),
        'min_samples_split' : st.randint(2, 8),
        'min_samples_leaf' : st.randint(1, 8),
        'bootstrap' : [True, False],
        'criterion' : ["gini", "entropy"]
}

n_iter_search = 20
RSCV_rf = RandomizedSearchCV(rf, param_dist, scoring='accuracy', n_iter=n_iter_search, cv=10)


# In[ ]:


start = time()
RSCV_rf.fit(X_train, y_train)

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(RSCV_rf.cv_results_)


# In[ ]:


y_pred = RSCV_rf.predict(X_test)


# In[ ]:


conf_matrix = metrics.confusion_matrix(y_test,y_pred)
acc = metrics.accuracy_score(y_test, y_pred)*100
print(conf_matrix)
print('\nAccuracy : %0.2f %%\n' % acc)
print(classification_report(y_test, y_pred))


# In[ ]:


importances = RSCV_rf.best_estimator_.feature_importances_


# In[ ]:


std = np.std([tree.feature_importances_ for tree in RSCV_rf.best_estimator_], axis=0)
indices = np.argsort(importances)[::-1]


# In[ ]:


print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# In[ ]:


plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.grid(True)


# **4. RandomForest  - RandomizedSearchCV**
# * Accuracy = 83.5 %

# 

# In[ ]:


# 5. GradientBoostingClassifier / RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()

param_dist = {
        'n_estimators': st.randint(10, 100),
        'max_depth': [3, None],
        'min_samples_leaf': st.randint(1, 5)
}

n_iter_search = 20
RSCV_gb = RandomizedSearchCV(gb, param_dist, scoring='accuracy', n_iter=n_iter_search, cv=10)


# In[ ]:


start = time()
RSCV_gb.fit(X_train, y_train)

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(RSCV_gb.cv_results_)


# In[ ]:


y_pred = RSCV_gb.predict(X_test)


# In[ ]:


conf_matrix = metrics.confusion_matrix(y_test, y_pred)
acc = metrics.accuracy_score(y_test, y_pred)*100
print(conf_matrix)
print('\nAccuracy : %0.2f %%\n' % acc)
print(classification_report(y_test, y_pred))


# In[ ]:


importances = RSCV_gb.best_estimator_.feature_importances_


# In[ ]:


indices = np.argsort(importances)[::-1]


# In[ ]:


print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# In[ ]:


plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r",align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.grid(True)


# **5. GradientBoosting - RandomizedSearchCV **
# * Accuracy = 86.5 %

# 

# In[ ]:


# XGBoost / RandomizedSearchCV
import xgboost as xgb

# you should consider parma : 
# 1. booster [default=gbtree]
# 2. num_feature [default=MAX]
# 3. alpha, lambda
param = {
        'n_estimators' : 100,
        'max_depth' : 5,
        'learning_rate' : 0.1,
        'colsample_bytree' : 0.8,
        'subsample' : 0.8,
        'gamma' : 0,
        'min_child_weight' : 1
}


# In[ ]:


xgtrain = xgb.DMatrix(X_train, label=y_train)
xgtest = xgb.DMatrix(X_test, label=y_test)


# In[ ]:


num_rounds = 100
watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
xgb_model = xgb.train(param, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)


# In[ ]:


y_pred = xgb_model.predict(xgtest)


# In[ ]:


xgb_model.attributes()


# In[ ]:


xgb_model.get_score()


# In[ ]:


xgb.plot_importance(xgb_model)


# In[ ]:


predictions = [round(value) for value in y_pred]
acc =  metrics.accuracy_score(y_test, predictions)*100
print('\nAccuracy: %.2f %%\n' % acc)


# **6. XGBoost (Set param)  **
# * Accuracy = 86.25 %

# 

# In[ ]:


# Use RandomizedSearchCV
param_dist = {  
        'n_estimators' : st.randint(20, 100),
        'max_depth' : st.randint(5, 20),
        'learning_rate' : st.uniform(0.05, 0.2),
        'colsample_bytree' : st.beta(10, 1),
        'subsample' : st.beta(10, 1),
        'gamma' : st.uniform(0, 10),
        'min_child_weight' : st.expon(0, 10)
}


# In[ ]:


xgbc = xgb.XGBClassifier(nthreads=-1)


# In[ ]:


n_iter_search = 20
RSCV_xgbc = RandomizedSearchCV(xgbc, param_dist, scoring='accuracy', n_iter=n_iter_search, cv=10)


# In[ ]:


RSCV_xgbc.fit(X_train, y_train)  


# Error : /opt/conda/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
#   if diff:
# 
# There is no problem. you can ignore this error msg -> refer : https://github.com/scikit-learn/scikit-learn/pull/9816

# In[ ]:


RSCV_xgbc.best_params_


# In[ ]:


RSCV_xgbc.best_score_


# In[ ]:


y_pred = RSCV_xgbc.predict(X_test)


# In[ ]:


conf_matrix = metrics.confusion_matrix(y_test,y_pred)
predictions = [round(value) for value in y_pred]
acc =  metrics.accuracy_score(y_test, predictions)*100
print(conf_matrix)
print('\nAccuracy: %.2f %%\n' % acc)
print(classification_report(y_test, y_pred))


# **6. XGBoost - RandomizedSearchCV **
# * Accuracy = 87.75 %

# 

# **Thank you for reading My RedWineAnalysis !!**
# 
# If this note was helpful? **Please Upvote**
