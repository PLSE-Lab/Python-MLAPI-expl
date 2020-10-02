#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib import rcParams
import seaborn as sns
import re
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


#from sklearn.cross_validation import KFold, cross_val_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import roc_curve, accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV

from sklearn import ensemble
from sklearn import tree

rcParams['figure.figsize'] = (8, 5)


# In[ ]:


testing = pd.read_csv('/kaggle/input/titanic/test.csv')
train = pd.read_csv('/kaggle/input/titanic/train.csv')

target = 'Survived'

test = testing.copy()

#checking data type and null values, overview of dataset
test.info()
print('-'*70)
train.info()
print('-'*70)
train.tail(10)


# In[ ]:


#feature engineering
def title(i):
    i['Title'] = i.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
    i['Title'] = i['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
    i['Title'] = i['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                                'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
title(train)
title(test)
#---------------------------------------------------------------------------------------------------------------------------------------

def fillna_median_age(i, u):
    i[u] = i.groupby(['Title'])[u].apply(lambda x: x.fillna(x.median()))
    
fillna_median_age(train,'Age')
fillna_median_age(test,'Age')
#---------------------------------------------------------------------------------------------------------------------------------------

def fillna_median_fare(i, u):
    i[u] = i.groupby(['Pclass', 'Sex'])[u].apply(lambda x: x.fillna(x.median()))
    
fillna_median_fare(test,'Fare')
#---------------------------------------------------------------------------------------------------------------------------------------

train['Embarked'] = train['Embarked'].fillna("S")
#---------------------------------------------------------------------------------------------------------------------------------------

def family_size(x):
    x['Family'] =  x["Parch"] + x["SibSp"]

family_size(train)
family_size(test)


# In[ ]:


train.describe()


# In[ ]:


# Preparing features for analysis
dummy_features = ['Sex','Embarked','Title']
drop_features = ['PassengerId', 'Ticket', 'Name', 'Cabin','Parch','SibSp']
    
train = pd.concat([train, pd.get_dummies(train[dummy_features])], axis = 1, sort = False)
train.drop(columns = train[dummy_features], inplace = True)
train.drop(columns = train[drop_features], inplace = True)

test = pd.concat([test, pd.get_dummies(test[dummy_features])], axis = 1, sort = False)
test.drop(columns = test[dummy_features], inplace = True)
test.drop(columns = test[drop_features], inplace = True)

train.tail()


# In[ ]:


cor = train.corr()
cor_target = abs(cor[target])

pyplot.figure(figsize=(10,8))
sns.heatmap(cor, annot = True);

# #Selecting highly correlated features
# relevant_features = cor_target[cor_target>=0.0]
# print(relevant_features)


# In[ ]:


#last check for NaN values in dataset and check if column amount is the same in both datasets
train.info()
print('-'*70)
test.info()


# In[ ]:


# #data normalization
# x_df = (x_df - np.min(x_df)) / (np.max(x_df)-np.min(x_df))
# test = (test - np.min(test)) / (np.max(test)-np.min(test))

# print(x_df.mean(axis=0))
# print("-"*50)
# print(test.mean(axis=0))


# In[ ]:


# Separating target column from other features
y = train[target]
x = train.drop(columns = target)

# Train and Test dataset split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)


# In[ ]:


def ROC_curve_plot(model):
    # Area under the curve probability score
    probability = model.predict_proba(x_test)
    probability = probability[:, 1]
    auc = roc_auc_score(y_test, probability)
    print('AUC value: %.3f' % auc)

    # AUC plot
    pyplot.style.use('default')
    fpr, tpr, thresholds = roc_curve(y_test, probability)
    pyplot.figure(figsize=(6,4))
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    pyplot.plot(fpr, tpr, color = 'tab:orange')
    pyplot.show()


# In[ ]:


# random forest model hyper-tuned

RF = ensemble.RandomForestClassifier()
RF_params = {
          'n_estimators':[n for n in range(40,140,20)],
          'max_depth':[n for n in range(2, 8)],
          #'min_samples_leaf': [n for n in range(2, 6, 2)],
          'random_state' : [42]
            }

RF_model = GridSearchCV(RF, param_grid = RF_params, cv = 5, n_jobs = -1)
RF_model.fit(x_train, y_train)

print("Best Hyper Parameters: ", RF_model.best_params_)
print("Best Score: " + '%.3f' % RF_model.best_score_)

RF_predictions = RF_model.predict(x_test)
RF_accuracy = accuracy_score(y_test, RF_predictions)
print("RF accuracy: " + '%.3f' % RF_accuracy)

ROC_curve_plot(RF_model)


# In[ ]:


#gradient boosting tree model hyper-tuned

GBT = ensemble.GradientBoostingClassifier()
GBT_params = {
          'n_estimators':[n for n in range(180, 220, 5)],
          'max_depth':[n for n in range(2, 4)],
          #'min_samples_leaf': [n for n in range(2, 6, 2)],
          'random_state' : [42]
             }

GBT_model = GridSearchCV(GBT, param_grid = GBT_params, cv = 8, n_jobs = -1)
GBT_model.fit(x_train, y_train)
print("Best Hyper Parameters: ",GBT_model.best_params_)
print("Best Score: " + '%.3f' % GBT_model.best_score_)

GBT_predictions = GBT_model.predict(x_test)
GBT_accuracy = accuracy_score(y_test, GBT_predictions)
print("GBT accuracy: " + '%.3f' % GBT_accuracy)

ROC_curve_plot(GBT_model)


# In[ ]:


#comparing model predictions to org. submission file
b = '\033[1m'
ub = '\033[0m'

def conf_matrix(x, y):
    prediction = x.predict(x_test)
    CM = confusion_matrix(y_test, prediction)
    CM_rel = CM / CM.astype(np.float).sum(axis=1)


    pyplot.figure(figsize=(6,5))
    sns.heatmap(CM_rel, annot = True, fmt='.2f')
    pyplot.ylabel('Actual')
    pyplot.xlabel('Predicted')
    pyplot.title(y);
    
    print(ub + y, ' confusion matrix results:\n', CM, '\n')
    
    pred = accuracy_score(y_test, prediction)
    print(ub + y + " accuracy: " + b +   '%.3f' % pred)
    
    
conf_matrix(RF_model, 'Random Forest')   
print('='*50,'\n')
conf_matrix(GBT_model, 'Gradient Boost')


# In[ ]:


predict_RF = RF_model.predict(test)
predict_GBT = GBT_model.predict(test)

submit_RF = pd.DataFrame({'PassengerId':testing['PassengerId'],'Survived':predict_RF})
submit_GBT = pd.DataFrame({'PassengerId':testing['PassengerId'],'Survived':predict_GBT})


#creating submission file
filename_RF = 'Titanic Prediction RF.csv'
submit_RF.to_csv(filename_RF,index=False)
print('Saved file: ' + filename_RF)

filename_GBT = 'Titanic Prediction GBT.csv'
submit_GBT.to_csv(filename_GBT,index=False)
print('Saved file: ' + filename_GBT)

