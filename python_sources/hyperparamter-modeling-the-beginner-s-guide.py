#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning Beginner Guide - Topics
# 
# - Data Analysis
# - Data Scaling
# - Data Modeling
# - Hyper-parameter Tuning.
# - Cross Validation
# - Predictions

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import sklearn
#sklearn.metrics.SCORERS.keys()
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
print(data.shape)
data.tail()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


# In[ ]:


data.info()


# In[ ]:


data.isnull().any()


# In[ ]:


X = data.iloc[:,:-1] # Independent variables
y = data['Outcome'] # Dependent Variables


# ## Data Analysis

# In[ ]:


fig = plt.figure(figsize=(18,16))
for index,col in enumerate(X):
    plt.subplot(5,4,index+1)
    sns.distplot(X.loc[:,col],kde= False)
fig.tight_layout(pad=1.0)


# In[ ]:


plt.figure(figsize=(18,16))
for index,col in enumerate(X):
    plt.subplot(5,4,index+1)
    sns.boxplot(y = col, data= X)
fig.tight_layout(pad=1.0)


# ### Removing outliers from the Independent variables

# In[ ]:


data = data.drop(data[data['Pregnancies']>11].index)
data = data.drop(data[data['Glucose']<30].index)
data = data.drop(data[data['BloodPressure']>110].index)
data = data.drop(data[data['BloodPressure']<20].index)
data = data.drop(data[data['SkinThickness']>80].index)
data = data.drop(data[data['BMI']>55].index)
data = data.drop(data[data['BMI']<10].index)
data = data.drop(data[data['DiabetesPedigreeFunction']>1.6].index)
data = data.drop(data[data['Insulin']>400].index)
data = data.drop(data[data['Age']>80].index)


# In[ ]:


data.shape


# In[ ]:


plt.figure(figsize=(11,10))
correlation = X.corr()
sns.heatmap(correlation,linewidth = 0.7,cmap = 'Blues',annot = True)


# In[ ]:


X = X.loc[data.index]
y = y.loc[data.index]


# ## Data Scaling

# from sklearn.preprocessing import RobustScaler
# cols = X.columns
# transformer = RobustScaler().fit(X[cols])
# X[cols] = transformer.transform(X[cols])
# 
# ### Note it is not necessary to Scale the data for boosting method

# ## Data Modeling

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.25,random_state = 100)


# In[ ]:


from sklearn.metrics import log_loss,accuracy_score,confusion_matrix,f1_score,recall_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score,StratifiedKFold,RandomizedSearchCV
xgb = XGBClassifier(booster ='gbtree',objective ='binary:logistic')


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

param_lst = {
    'learning_rate' : [0.01, 0.1, 0.15, 0.3, 0.5,0.4],
    'n_estimators' : [100, 500, 1000,1500,2000],
    'max_depth' : [2,3,5, 6,8, 9],
    'min_child_weight' : [1, 5, 10],
    'reg_alpha' : [0.001, 0.01, 0.1],
    'reg_lambda' : [0.001, 0.01, 0.1],
    'colsample_bytree' : [0.3,0.4,0.5,0.7],
    'gamma' : [0.0,0.1,0.2,0.3,0.4]
}

xgb_tuning = RandomizedSearchCV(estimator = xgb, param_distributions = param_lst ,
                          n_iter = 5,
                        cv =6)
       
xgb_search = xgb_tuning.fit(X_train,y_train,
                           early_stopping_rounds = 5,
                           eval_set=[(X_val,y_val)],
                           verbose = False)

##hyperparameter tuning

best_param = xgb_search.best_params_
xgb = XGBClassifier(**best_param)
print(best_param)


# In[ ]:


xgb_search.best_estimator_


# In[ ]:


y_pred = xgb_search.predict(X_val)
score0 = accuracy_score(y_pred,y_val)
#print(round(score0*100,4))
print('Score: {}%'.format(round(score0*100,4)))


# In[ ]:


acc_scores1_xgb =  cross_val_score(xgb_search,X,y,n_jobs=5,
                                 cv = StratifiedKFold(n_splits=10),
                                 scoring = 'accuracy')
acc_scores1_xgb


# In[ ]:


print(acc_scores1_xgb.mean())


# In[ ]:


from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn import metrics

fpr, tpr, threshold = metrics.roc_curve(y_val, y_pred)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'red', label = 'ROC AUC score = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'b--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Using Cross Validation with Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(X_train,y_train)


# In[ ]:


log_scores_logi = -1 * cross_val_score(lr, X, y,
                              cv=5,
                              scoring='neg_log_loss')
acc_scores1_logi =  cross_val_score(lr,X,y,
                                 cv = 5,
                                 scoring = 'accuracy')
f_score_logi =  cross_val_score(lr,X,y,
                                 cv = 5,
                                 scoring = 'f1')


# In[ ]:


print("log_loss scores:\n", log_scores_logi)
print("Accuracy scores:\n", acc_scores1_logi)
print("f1_score scores:\n", f_score_logi)


# In[ ]:


print(acc_scores1_logi.mean())


# ### If you do like my work and found some useful insights from it, please Upvote and comment your thoughts below. 
# 
# 
# - Some References I took from:
#  - [Tuning](https://www.kaggle.com/angqx95/data-science-workflow-top-2-with-tuning)
#  - [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
#  - [XGBClassifier](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
