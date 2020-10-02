#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# 
# 
# 
# > **Learn with other Kaggle Users Compettition**
# 
# 
# 

# In[ ]:


#import libraries

import pandas as pd
import pandas_profiling as pp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os



df_train = pd.read_csv('../input/learn-together/train.csv')
df_test = pd.read_csv('../input/learn-together/test.csv')





for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train.head()


# In[ ]:


df_train.columns


# In[ ]:


#split data into training/validation
#our target is cover_Type
from sklearn.model_selection import train_test_split
lst = ['Id', 'Cover_Type']

X = df_train.drop(lst, axis=1)
y = df_train.Cover_Type
test_X = df_test.drop('Id', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.3,
                                                   random_state=0)

print('X_train: ',X_train.shape)
print('X_test: ',X_test.shape)
print('y_train: ',y_train.shape)
print('y_test: ',y_test.shape)


# In[ ]:


#model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, cross_val_score
models = []
models.append(( ' LR ' , LogisticRegression()))
models.append(( ' LDA ' , LinearDiscriminantAnalysis()))
models.append(( ' KNN ' , KNeighborsClassifier()))
models.append(( ' NB ' , GaussianNB()))
models.append(( ' SVM ' , SVC()))

results = []
names = []

for name, model in models:
    Kfold = KFold(n_splits=10, random_state=0)
    cv_results = cross_val_score(model, X_train, y_train, cv=Kfold, scoring= 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std());
    print(msg)


# In[ ]:


#improve Performance
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
models = []
models.append(( 'Adab' , AdaBoostClassifier()))
models.append(( 'Bagging' , BaggingClassifier()))
models.append(( 'GBC' , GradientBoostingClassifier()))
models.append(( 'RF' , RandomForestClassifier()))


results = []
names = []

for name, model in models:
    Kfold = KFold(n_splits=10, random_state=0)
    cv_results = cross_val_score(model, X_train, y_train, cv=Kfold, scoring= 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std());
    print(msg)


# In[ ]:


# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer
# Tuning
from sklearn.model_selection import GridSearchCV

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)


model = RandomForestClassifier()
param_grid = { 
    'n_estimators': [10,20,50,100],
    'max_features': ['auto', 'sqrt', 'log2']
}

kfold = KFold(n_splits=10, random_state=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))



# In[ ]:


scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test =scaler.transform(X_test)
RF = RandomForestClassifier(n_estimators=100, max_features='auto').fit(X_train,y_train)
y_pred = RF.predict(test_X)


# In[ ]:


y_pred_1 = RF.predict(test_X)
sub = pd.DataFrame({'ID': df_test.Id,
                       'TARGET': y_pred_1})
sub.to_csv('submission.csv')

