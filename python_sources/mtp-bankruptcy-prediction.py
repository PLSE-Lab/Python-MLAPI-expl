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
files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        files+= [os.path.join(dirname, filename)]

# Any results you write to the current directory are saved as output.


# In[ ]:


files


# In[ ]:


# df = pd.read_csv(r'/kaggle/input/bankruptcy-attr/data/5year.arff')
# print('class 0:',len(df[df['class']==0]))
# print('class 1:',len(df[df['class']==1]))
# df


# In[ ]:


files


# In[ ]:


from scipy.io import arff
import pandas as pd

data = arff.loadarff(files[3])
df = pd.DataFrame(data[0])
df.head()


# In[ ]:


df= df.fillna(0)


# In[ ]:


X = df.drop(['class'],axis=1)
X = np.asarray(X)


# In[ ]:


df['class'] = df['class'].apply(lambda x:int(x))
y =df['class'].values


# AdaBoost Model

# In[ ]:


from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection

abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# ada_model = abc.fit(X_train, y_train)
# y_ada_pred = ada_model.predict(X_test)

# print(classification_report(y_test,y_ada_pred))
seed =10
# print("Accuracy:%.2f%%"%(accuracy_score(y_test,y_ada_pred)*100))
kfold = model_selection.StratifiedKFold(n_splits=10,random_state=seed)
results = model_selection.cross_val_score(abc,X,y,cv=kfold,scoring = 'roc_auc')


# In[ ]:


print('AdaBoost Model AUC Score:',results.mean())


# XGBoost Model

# In[ ]:


from xgboost import XGBClassifier

xgb = XGBClassifier()
# xgb_model = xgb.fit(X_train,y_train)
# y_xgb_pred = xgb_model.predict(X_test)
kfold = model_selection.StratifiedKFold(n_splits=10,random_state=seed)
results_xgb = model_selection.cross_val_score(xgb,X,y,cv=kfold,scoring = 'roc_auc')
print('XGBoost Model AUC Score',results_xgb.mean())


# RandomForest Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300,
                                max_features="sqrt",
                                criterion="gini",
                                min_samples_leaf=5,
                                class_weight="balanced")
kfold = model_selection.StratifiedKFold(n_splits=10,random_state=seed)
results_rf = model_selection.cross_val_score(rf,X,y,cv=kfold,scoring = 'roc_auc')
print('RandomForest AUC Score',results_rf.mean())


# SVM Model

# 1.Linear Kernel

# In[ ]:


from sklearn import svm
linear_clf = svm.SVC(kernel='linear')
kfold = model_selection.StratifiedKFold(n_splits=10,random_state=seed)
results_lsvm = model_selection.cross_val_score(linear_clf,X,y,cv=kfold,scoring = 'roc_auc')
print('SVM Linear Kernel Model AUC Score',results_lsvm.mean())


# 2.Radial Basis Function

# In[ ]:


rbf_clf = svm.SVC(kernel = 'RBF')
kfold = model_selection.StratifiedKFold(n_splits=10,random_state=seed)
results_rbf = model_selection.cross_val_score(rbf_clf,X,y,cv=kfold,scoring = 'roc_auc')
print('SVM Linear Kernel Model AUC Score',results_rbf.mean())


# ANN

# In[ ]:


from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from keras.constraints import maxnorm
from keras.optimizers import SGD
import keras.metrics as km


# In[ ]:


def mlp_baseline():
    model = Sequential()
    model.add(Dropout(0.2,input_shape=(64,)))
    model.add(Dense(70,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dense(35,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
    sgd = SGD(lr=0.1,momentum=0.9,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics = ['accuracy'])
    return model


# In[ ]:


# X.shape
seed=10
np.random.seed(seed)
estimators = []
estimators.append(('Standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(mlp_baseline,nb_epoch=100,batch_size=5,verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits = 10,shuffle=True,random_state=seed)

results_ann = cross_val_score(pipeline,X,y,cv=kfold)
print('ANN Accuracy:',results_ann.mean()*100,'%')

