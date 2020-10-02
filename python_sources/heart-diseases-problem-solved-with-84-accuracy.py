#!/usr/bin/env python
# coding: utf-8

# # Heart Diseases problem

# **Before we start:**
# 
# If you like my work, please upvote this kernel as it will keep me motivated to do more in the future and share the kernel with others so we can all benefit from it .

# ## Loading libraries

# In[ ]:


#loading_all libraries
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
from scipy import stats
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import csv
import os


# In[ ]:


warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading dataset using pd.read_csv

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")


# In[ ]:


df_train.head()


# In[ ]:


df_train.dtypes


# In[ ]:


#function for missing data
def missing_data(df_train):
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return(missing_data.head(20))


# In[ ]:


missing_data(df_train)


# ## Correlation Matrix

# In[ ]:


#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corrmat, vmax=.8, square=True);


# ## Data Visulization

# In[ ]:


plt.figure(figsize=(25,20))
sns.factorplot(data=df_train,x='target',y='age',hue='sex')


# In[ ]:


plt.figure(figsize=(15,10))
sns.relplot(x='trestbps', y='chol', data=df_train,
            kind='line', hue='fbs', col='sex')


# In[ ]:


plt.figure(figsize=(15,10))
sns.catplot(x='cp',y='oldpeak',data=df_train,hue='target',height=5,aspect=3,kind='box')
plt.title('boxplot')


# In[ ]:


plt.figure(figsize=(15,15))
sns.relplot(x='restecg', y='thalach', data=df_train,
            kind='line')


# In[ ]:


plt.figure(figsize=(15,7))
sns.countplot(x='slope',hue='sex',data=df_train,order=df_train['thal'].value_counts().sort_values().index);


# In[ ]:


sns.set()
cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
sns.pairplot(df_train[cols], size = 2.5)
plt.show()


# ## train test split

# In[ ]:


dependent_all=df_train['target']
independent_all=df_train.drop(['target'],axis=1)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(independent_all,dependent_all,test_size=0.3,random_state=100)


# ## Logistic regression

# In[ ]:


log =LogisticRegression()
log.fit(x_train,y_train)


# In[ ]:


#model on train using all the independent values in df
log_prediction = log.predict(x_train)
log_score= accuracy_score(y_train,log_prediction)
print('Accuracy score on train set using Logistic Regression :',log_score)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, log_prediction)
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_train,log_prediction)
print("AUC on train using Logistic Regression :",metrics.auc(fpr, tpr))
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_train, log_prediction)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
from sklearn.metrics import recall_score
print('recall_score on train set :',recall_score(y_train, log_prediction))
from sklearn.metrics import f1_score
print('F1_sccore on train set :',f1_score(y_train, log_prediction))


# In[ ]:


#model on train using all the independent values in df
log_prediction = log.predict(x_test)
log_score= accuracy_score(y_test,log_prediction)
print('accuracy score on test using Logisitic Regression :',log_score)


# In[ ]:


confusion_matrix(y_test, log_prediction)
fpr, tpr, thresholds = metrics.roc_curve(y_test,log_prediction)
print("AUC on test using Logistic Regression :",metrics.auc(fpr, tpr))
average_precision = average_precision_score(y_test, log_prediction)
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
print('recall_score on test set :',recall_score(y_test, log_prediction))
print('F1_sccore on test set :',f1_score(y_test, log_prediction))


# ## Kfold cross validation

# In[ ]:


from sklearn.model_selection import cross_val_score
lr = LogisticRegression()
scores = cross_val_score(lr, x_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# ## Xgboost 

# In[ ]:


xgboost = xgb.XGBClassifier(max_depth=3,n_estimators=300,learning_rate=0.001)


# In[ ]:


xgboost.fit(x_train,y_train)


# In[ ]:


#XGBoost model on the train set
XGB_prediction = xgboost.predict(x_train)
XGB_score= accuracy_score(y_train,XGB_prediction)
print('accuracy score on train using XGBoost ',XGB_score)


# In[ ]:


from sklearn import metrics
print(confusion_matrix(y_train, XGB_prediction))
fpr, tpr, thresholds = metrics.roc_curve(y_train,XGB_prediction)
print("AUC on train using XGBClassifiers:",metrics.auc(fpr, tpr))

average_precision = average_precision_score(y_train, XGB_prediction)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
print('recall_score on train set :',recall_score(y_train, XGB_prediction))
print('F1_sccore on train set :',f1_score(y_train, XGB_prediction))


# In[ ]:


#XGBoost model on the test
XGB_prediction = xgboost.predict(x_test)
XGB_score= accuracy_score(y_test,XGB_prediction)
print('accuracy score on test using XGBoost :',XGB_score)


# In[ ]:


from sklearn import metrics
print(confusion_matrix(y_test, XGB_prediction))
fpr, tpr, thresholds = metrics.roc_curve(y_test,XGB_prediction)
print("AUC on test using XGBClassifiers:",metrics.auc(fpr, tpr))

average_precision = average_precision_score(y_test, XGB_prediction)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
print('recall_score on test set :',recall_score(y_test, XGB_prediction))
print('F1_sccore on test set :',f1_score(y_test, XGB_prediction))


# ## Kfold cross validation

# In[ ]:


xg = xgb.XGBClassifier()
scores = cross_val_score(xg, x_test, y_test, cv=5, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# ## Random forest

# In[ ]:


rfc2=RandomForestClassifier(n_estimators=100)
rfc2.fit(x_train,y_train)


# In[ ]:


#model on train using all the independent values in df
rfc_prediction = rfc2.predict(x_train)
rfc_score= accuracy_score(y_train,rfc_prediction)
print('accuracy Score on train using RandomForest :',rfc_score)


# In[ ]:


from sklearn import metrics
print(confusion_matrix(y_train, rfc_prediction))
fpr, tpr, thresholds = metrics.roc_curve(y_train,rfc_prediction)
print("AUC on train using RandomForest :",metrics.auc(fpr, tpr))

average_precision = average_precision_score(y_train, rfc_prediction)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
print('recall_score on train set :',recall_score(y_train, rfc_prediction))
print('F1_sccore on train set :',f1_score(y_train, rfc_prediction))


# In[ ]:


#model on test using all the indpendent values in df
rfc_prediction = rfc2.predict(x_test)
rfc_score= accuracy_score(y_test,rfc_prediction)
print('accuracy score on test using RandomForest ',rfc_score)


# In[ ]:



print(confusion_matrix(y_test, rfc_prediction))
fpr, tpr, thresholds = metrics.roc_curve(y_test,rfc_prediction)
print("AUC on test using RandomForest :",metrics.auc(fpr, tpr))

average_precision = average_precision_score(y_test, rfc_prediction)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
print('recall_score on test set :',recall_score(y_test, rfc_prediction))
print('F1_sccore on test set :',f1_score(y_test, rfc_prediction))


# ## Kfold cross validation 

# In[ ]:


lr = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(lr, x_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# # GridSearchCV [RandomForestClassifier]

# In[ ]:


clf = RandomForestClassifier()
grid_values = {'max_features':['auto','sqrt','log2'],'max_depth':[None, 10, 5, 3, 1],
              'min_samples_leaf':[1, 5, 10, 20, 50]}
grid_clf = GridSearchCV(clf, param_grid=grid_values, cv=10, scoring='accuracy')
grid_clf.fit(x_train, y_train)


# In[ ]:


grid_clf.best_params_


# In[ ]:


clf = RandomForestClassifier().fit(x_train, y_train)


# In[ ]:


y_pred = clf.predict(x_test)


# In[ ]:


print('Training Accuracy :: ', accuracy_score(y_train, clf.predict(x_train)))
print('Test Accuracy :: ', accuracy_score(y_test, y_pred))


# ## Conclusion
from all above caluculation we can conclude that xgboost algorithm is best this problem :)