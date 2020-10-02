#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Standard plotly imports
#import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
#import cufflinks
#import cufflinks as cf
import plotly.figure_factory as ff

# Using plotly + cufflinks in offline mode
init_notebook_mode(connected=True)
#cufflinks.go_offline(connected=True)

# Preprocessing, modelling and evaluating
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb

## Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial

import os
import gc
print(os.listdir("../input"))


# In[ ]:


df_train = pd.read_csv("../input/trainsample_SEG5_V1.csv")
df_train.head()


# In[ ]:


df_train.info()


# In[ ]:


df_train = df_train.drop(['event1'],axis=1)


# 1. No missing values
# 2. Removed Event1 variable 

# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot(x=df_train['event'], data=df, order=[1,0])


# In[ ]:


df_train.columns


# In[ ]:


categorical_var=['event','TOP_PERSO', 'TOP_IMMO', 'TOP_CAV',
       'TOP_GSM', 'TOP_ASSV', 'TOP_CB_VISA', 'TOP_CB_BUSI', 'TOP_CB_PREMIER',
       'TOP_CB_INFI', 'TOP_CB_MASTER', 'TOP_CONV_M_1', 'top_clot_cav12',
       'top_clot_pea12', 'top_clot_IMMO12', 'top_clot_PERSO12', 'top_staff',
       'TOP_PEA', 'TOP_TITRE', 'TOP_INV', 'top_cars', 'top_carte',
       'TOP_PAssport', 'Contact_3mois', 'Contact_6mois', 'Contact_9mois',
       'Contact_12mois']
categorical_var #List of categorical variable


# In[ ]:


#Function to plot Graph of each attribute vs target value
def PlotAttributeVsTarget(df,target):
    for y in df.columns:
        if y!= target:
            plt.figure(figsize=(12,5))
            #sns.scatterplot(x=df[y],y=df[target])
            sns.countplot(x=df[target], hue=df[y], data=df, order=[1,0])
    
PlotAttributeVsTarget(df_train_cat,'event')


# In[ ]:


x = df_train.drop(['event'],axis=1)
y = df_train['event']


# In[ ]:


from imblearn.over_sampling import SMOTE
print("Before OverSampling, counts of label '1': {}".format(sum(y==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y==0)))

sm = SMOTE(random_state=2)
X_res, y_res = sm.fit_sample(x, y.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_res==0)))


# In[ ]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=0)


# In[ ]:


X_train.columns()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


xg = xgb.XGBClassifier(colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100)


# In[ ]:


xg.fit(X_train,y_train)
y_pred = xg.predict(X_test)
print('Accuracy of xgb classifier on test set: {:.2f}'.format(xg.score(X_test, y_test)))


# In[ ]:


xgb.plot_importance(xg)
plt.figure(figsize=(12,5))
plt.show()


# In[ ]:


f_imp = xg.get_booster().get_score(importance_type='gain')
f_imp


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))


# In[ ]:


from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
thresholds = sort(xg.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(xg, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = xgb.XGBClassifier(colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100)
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
    #confusion_matrix = confusion_matrix(y_test, y_pred)
    #print(classification_report(y_test, y_pred))
    #print("F1 Score: {}".format(f1_score(y_true,y_pred)))


# In[ ]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator1,estimator2, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    f, ((ax1, ax2) = plt.subplots(2,2, figsize=(20,14), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
    # First Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")
    
    # Second Estimator 
    train_sizes, train_scores, test_scores = learning_curve(
        estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax2.set_title("Knears Neighbors Learning Curve", fontsize=14)
    ax2.set_xlabel('Training size (m)')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    ax2.legend(loc="best")
    return plt


# In[ ]:



cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
plot_learning_curve(xg,logreg, X_train, y_train, (0.87, 1.01), cv=cv, n_jobs=4)


# In[ ]:




