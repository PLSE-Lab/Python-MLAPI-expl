#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV


# In[ ]:


df = pd.read_csv('../input/creditcard.csv')
df.head()


# whether imbalance? 0:non-fraud, 1:fraud

# In[ ]:


df['Class'].value_counts()


# In[ ]:


print("No Fruads",round(df["Class"].value_counts()[0]/len(df)*100,2),'% of the dataset')
print("Fruads",round(df["Class"].value_counts()[1]/len(df)*100,2),'% of the dataset')


# In[ ]:


import pandas_profiling
pandas_profiling.ProfileReport(df)


# Any missing value? Outlier?

# In[ ]:


df.isna().sum()


# train data

# In[ ]:


y=df['Class']
X=df.drop(['Class'], axis=1)

X_train, X_test, y_train, y_test=train_test_split(X, y,test_size=0.2, random_state=0)
print('X_train.shape:', X_train.shape)
print('y_train.shape:', y_train.shape)


# In[ ]:


X_train.head()


# Normalization

# In[ ]:


scaler=preprocessing.MinMaxScaler().fit(X_train[['Time','Amount']])
print(scaler.data_max_)


# Transform the training data and use them for the model training

# In[ ]:


X_train[['Time','Amount']]=scaler.transform(X_train[['Time','Amount']])


# In[ ]:


print(X_train[['Time','Amount']].head())


# before the prediction of the test data, apply the same scaler obtained from above X_test, not fitting a brandnew scaler on test set

# In[ ]:


X_test[['Time','Amount']]=scaler.transform(X_test[['Time','Amount']])


# Simple Logistic Regression, using default parameter

# In[ ]:


logreg=LogisticRegression()
#fit the model with data
logreg.fit(X_train, y_train)
#predict on test set
y_pred=logreg.predict(X_test)


# In[ ]:


cm=metrics.confusion_matrix(y_test,y_pred)
cmDF=pd.DataFrame(cm, columns=['pred_0','pred_1'],index=['true_0','true_1'])
print(cmDF)
print('recall=', float(cm[1,1])/(cm[1,0]+cm[1,1]))
print('precision=', float(cm[1,1])/(cm[1,1]+cm[0,1]))


# In[ ]:


# simple rf
classifier_RF=RandomForestClassifier(random_state=0)
classifier_RF.fit(X_train, y_train)

# predict class labels 0/1 for the test set
predicted=classifier_RF.predict(X_test)

#generate class probabilities
probs=classifier_RF.predict(X_test)

# generate evaluation metics
print('%s: %r' % ('accuracy_score is:', accuracy_score(y_test, predicted)))
#print('%s: %r' % ('roc_auc_score is:', roc_auc_score(y_test, probs[:, 1])))
print('%s: %r' % ('f1_score is:', f1_score(y_test, predicted)))

print('confusion_matrix is:')
cm=confusion_matrix(y_test, predicted)
cmDF=pd.DataFrame(cm, columns=['pred_0','pred_1'],index=['true_0','true_1'])
print(cmDF)
print('recall=', float(cm[1,1])/(cm[1,0]+cm[1,1]))
print('precision=', float(cm[1,1])/(cm[1,1]+cm[0,1]))


# SMOTE SAMPLING

# In[ ]:


smote=SMOTE(random_state=12)
x_train_sm, y_train_sm=smote.fit_sample(X_train, y_train)
unique, counts=np.unique(y_train_sm, return_counts=True)
print(np.asarray((unique, counts)).T)


# RF on smoted training data

# In[ ]:


# simple rf
classifier_RF_sm=RandomForestClassifier(random_state=0)
classifier_RF_sm.fit(X_train, y_train)

# predict class labels 0/1 for the test set
predicted_sm=classifier_RF.predict(X_test)

#generate class probabilities
probs_sm=classifier_RF.predict(X_test)

# generate evaluation metics
print('%s: %r' % ('accuracy_score_sm is:', accuracy_score(y_test, predicted)))
#print('%s: %r' % ('roc_auc_score_sm is:', roc_auc_score(y_test, probs[:, 1])))
print('%s: %r' % ('f1_score_sm is:', f1_score(y_test, predicted)))

print('confusion_matrix_sm is:')
cm_sm=confusion_matrix(y_test, predicted_sm)
cmDF_sm=pd.DataFrame(cm_sm, columns=['pred_0','pred_1'],index=['true_0','true_1'])
print(cmDF_sm)
print('recall or sens_sm=', float(cm_sm[1,1])/(cm_sm[1,0]+cm_sm[1,1]))
print('precision_sm=', float(cm_sm[1,1])/(cm_sm[1,1]+cm_sm[0,1]))


# parameter tuning by GridSearchCV

# In[ ]:


# evaluated metrics to be calculated for each combination of parameters and cv
scorers={
    'precison_score':make_scorer(precision_score),
    'recall_score':make_scorer(recall_score),
    'f1_score':make_scorer(f1_score,pos_label=1)
}

def grid_search_wrapper(model, parameters, refit_score='f1_score'):
    grid_search=GridSearchCV(model, parameters, scoring=scorers, refit=refit_score,
                            cv=3, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    #make predictions
    y_pred=grid_search.predict(X_test)
    y_prob=grid_search.predict_proba(X_test)[:, 1]
    
    print('Best params for{}'.format(refit_score))
    print(grid_search.best_params_)
    
    #confusion matrixs on test data
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    cm=confusion_matrix(y_test, y_pred)
    cmDF=pd.DataFrame(cm, columns=['pred_0','pred_1'],index=['true_0','true_1'])
    print(cmDF)
    
    print('recall=', float(cm[1,1])/(cm[1,0]+cm[1,1]))
    print('precision=', float(cm[1,1])/(cm[1,1]+cm[0,1]))
    
    return grid_search


# In[ ]:


# C: inverse of regularization strength, smaller values specify strong regularization
LRGrid={'C': np.logspace(-2,2,5),'penalty':['l1','l2']} #l1 lasso
#param_grid=
logRegModel=LogisticRegression(random_state=0)

grid_search_LR_f1=grid_search_wrapper(logRegModel, LRGrid, refit_score='f1_score')


# In[ ]:


#optimize on f1_score on RF
""""parameters={
    'n_estimators': [10, 150],
    'class_weight':[{0:1, 1:w} for w in [0.2, 1, 100]]
}
clf=RandomForestClassifier(random_state=0)
grid_search_rf_f1=grid_search_wrapper(clf, parameters, refit_score='f1_score')"""


# In[ ]:


""""best_rf_model_f1=grid_search_rf_f1.best_estimator_
best_rf_model_f1"""


# In[ ]:


"""
results_f1=pd.DataFrame(grid_search_rf_f1.cv_results_)
results_sortf1=results_f1.sort_values(by='mean_test_f1_score',ascending=False)
results_sortf1[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_f1_score', 'mean_train_precision_score', 'mean_train_recall_score', 'mean_train_f1_score', 'param_max_depth','param_class_weight','param_n_estimators']].round(3).head()
""""


# In[ ]:


#variable importance
#pd.DataFrame(best_rf_model_f1.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance',ascending=False)


# In[ ]:


# Optimize recall_score on RF
#grid_search_rf_recall=grid_search_wrapper(clf, parameters, refit_score='recall_score')


# In[ ]:


#best_RF_model_recall=grid_search_rf_recall.best_estimator_
#best_RF_model_recall

