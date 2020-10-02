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


train = pd.read_csv('/kaggle/input/learn-together/train.csv')


# In[ ]:


train.info()


# In[ ]:


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


# In[ ]:


train.head(5)


# In[ ]:


test=pd.read_csv('/kaggle/input/learn-together/test.csv')


# In[ ]:


test.describe().T


# In[ ]:


print(test.shape)
print(train.shape)


# In[ ]:


train['Cover_Type'].value_counts()


# In[ ]:


train.isnull().sum().sum()
test.isnull().sum().sum()


# In[ ]:


train.nunique()


# In[ ]:


X=train.drop(['Id','Soil_Type15','Soil_Type7','Cover_Type'],axis=1)


# In[ ]:


y=train['Cover_Type']


# In[ ]:


from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[ ]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# In[ ]:


results = []
names = []
scoring = 'accuracy'
for name, model in models:
  kfold = KFold(n_splits=10, random_state=7)
  cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)
# boxplot algorithm comparison
fig = pyplot.figure() 
fig.suptitle('Algorithm Comparison') 
ax = fig.add_subplot(111) 
pyplot.boxplot(results) 
ax.set_xticklabels(names) 
pyplot.show()


# In[ ]:


from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, 
                              ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# In[ ]:


models = []
models.append(('RFC', RandomForestClassifier(n_jobs =  -1, n_estimators = 500, max_features = 12, max_depth = 35, random_state = 1)))
models.append(('ADA', AdaBoostClassifier()))
models.append(('GrB', GradientBoostingClassifier()))
models.append(('EXTree', ExtraTreesClassifier()))
models.append(('Bag', BaggingClassifier()))
models.append(('LGBM', LGBMClassifier()))


# In[ ]:


results = []
names = []
scoring = 'accuracy'
for name, model in models:
  kfold = KFold(n_splits=10, random_state=7)
  cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)
# boxplot algorithm comparison
fig = pyplot.figure() 
fig.suptitle('Algorithm Comparison') 
ax = fig.add_subplot(111) 
pyplot.boxplot(results) 
ax.set_xticklabels(names) 
pyplot.show()


# In[ ]:


clf = XGBClassifier(n_estimators=500,colsample_bytree=0.9,max_depth=9,random_state=1,eta=0.2)
#clf.fit(x,y)


# In[ ]:


#kfold = KFold(n_splits=10, random_state=7)
cv_results = cross_val_score(clf, X, y, cv=3, scoring=scoring,n_jobs=3)


# In[ ]:


print(cv_results)


# In[ ]:


test.info()


# In[ ]:


ID = test['Id']
X_test = test.drop(['Id','Soil_Type7','Soil_Type15'],axis=1)


# '''doesnt work 
# 
# 
# 
# seed=1
# from keras.utils import to_categorical
# y_binary = to_categorical(y)
# #def baseline_model():
#   # create model
# model = Sequential()
# model.add(Dense(100, input_dim=52, kernel_initializer='normal', activation='relu')) 
# model.add(Dense(units=16, activation="relu"))
# model.add(Dense(8, kernel_initializer='normal', activation='sigmoid'))
# # Compile model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
# #    return model
# #estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
# #kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
# model.fit(X,y_binary,epochs=10,batch_size=100)
# #print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# '''

# In[ ]:


n_jobs = 8
seed=1
estimator=500
bag_clf = BaggingClassifier(n_estimators=estimator,
                            max_features = 15, max_depth = 35,
                            random_state=seed)

lg_clf = LGBMClassifier(n_estimators=estimator,
                        num_leaves=100,
                        verbosity=0,
                        random_state=seed,
                        n_jobs=n_jobs)

rf_clf = RandomForestClassifier(n_estimators=estimator,
                                min_samples_leaf=1,
                                verbose=0,
                                random_state=seed,
                                n_jobs=n_jobs)
et_clf = ExtraTreesClassifier(n_estimators=estimator,
                                min_samples_leaf=1,
                                verbose=0,
                                random_state=seed,
                                n_jobs=n_jobs)


final_model=[]
final_model.append(('RFC',RandomForestClassifier(n_jobs = n_jobs, n_estimators = estimator, max_features = 12, max_depth = 35, random_state = seed)))
final_model.append(('LGBM',LGBMClassifier(n_estimators=estimator,random_state=seed,n_jobs=n_jobs)))
final_model.append(('Bag', BaggingClassifier(n_estimators=estimator,random_state=seed,n_jobs=n_jobs)))


# In[ ]:


for name,model in final_model:
    model.fit(X,y)
    predict = model.predict(X_test)
    output = pd.DataFrame({'Id': ID,
                       'Cover_Type': predict})
    output.to_csv('submission_' + name + '.csv', index=False)


# ***
# Inspired by https://www.kaggle.com/phsheth/forestml-part-6-stacking-eval-selected-fets-2
# and
# https://www.kaggle.com/kwabenantim/forest-cover-stacking-multiple-classifiers
# ***

# In[ ]:


## Things to do
#Stacking classifier
#GridSearchCV to set the parameters
models=[]
scoring='accuracy'
models.append(('RFC', RandomForestClassifier(random_state = seed)))
models.append(('EXTree', ExtraTreesClassifier(random_state = seed)))
models.append(('Bag', BaggingClassifier(random_state = seed)))
models.append(('LGBM', LGBMClassifier(random_state = seed)))
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110,150],
    'max_features': [10,15,20,30,50],
    #'min_samples_leaf': [3, 4, 5],
    #'min_samples_split': [8, 10, 12],
    'n_estimators': [ 200, 300, 500,1000]
}
RFC=RandomForestClassifier(random_state = seed)
grid = GridSearchCV(estimator=RFC,n_jobs=16,param_grid =param_grid, scoring=scoring, cv=3,verbose=3)
grid_result = grid.fit(X, y)

#for name,model in models:
#    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=3)
#    grid_result = grid.fit(X, y)
#    print("%s -- Best: %f using %s" % (name,grid_result.best_score_, grid_result.best_params_))


# In[ ]:


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[ ]:


grid_result.best_estimator_


# In[ ]:


predict_grid_search = grid_result.best_estimator_.predict(X_test)


# In[ ]:


output = pd.DataFrame({'Id': ID,
                       'Cover_Type': predict_grid_search})
output.to_csv('submission_predict_grid_search.csv', index=False)


# In[ ]:


from IPython.display import FileLink
#FileLink(r'df_name.csv')


# In[ ]:


FileLink('submission_predict_grid_search.csv')


# In[ ]:


## Things to do
#Stacking classifier
#GridSearchCV to set the parameters
scoring = 'accuracy'
gridParams = {
    'learning_rate': [0.005,0.01],
    'n_estimators': [40,80,120,400,800],
    'num_leaves': [6,8,12,16],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
  #  'random_state' : [501], # Updated from 'seed'
    'colsample_bytree' : [0.65, 0.66],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4],
    }

LGBM=LGBMClassifier(random_state = seed)
grid = GridSearchCV(estimator=LGBM,n_jobs=16,param_grid =gridParams, scoring=scoring, cv=3,verbose=3)
grid_result = grid.fit(X, y)

#for name,model in models:
#    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=3)
#    grid_result = grid.fit(X, y)
#    print("%s -- Best: %f using %s" % (name,grid_result.best_score_, grid_result.best_params_))


# In[ ]:


predict_grid_search = grid_result.best_estimator_.predict(X_test)
output = pd.DataFrame({'Id': ID,
                       'Cover_Type': predict_grid_search})
output.to_csv('submission_predict_grid_search.csv', index=False)
from IPython.display import FileLink
FileLink(r'submission_predict_grid_search.csv')


# In[ ]:


grid_result.best_estimator_


# In[ ]:


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[ ]:


LGBM=LGBMClassifier(n_estimators=estimator,random_state=seed,n_jobs=n_jobs)


# In[ ]:


LGBM_new = LGBMClassifier(n_estimators=800,random_state=seed,n_jobs=16,reg_alpha=1, reg_lambda=1.2, subsample=0.7,
                          colsample_bytree= 0.66, learning_rate= 0.11, num_leaves= 16)


# In[ ]:


LGBM_new.fit(X,y)


# In[ ]:


predict_LGBM_new = LGBM_new.predict(X_test)
output = pd.DataFrame({'Id': ID,
                       'Cover_Type': predict_LGBM_new})
output.to_csv('submission_predict_LGBM_new.csv', index=False)
from IPython.display import FileLink
FileLink(r'submission_predict_LGBM_new.csv')


# In[ ]:


pip install mlxtend


# In[ ]:


from mlxtend.classifier import StackingCVClassifier


# In[ ]:


n_jobs = 8
seed=1
estimator=500
bag_clf = BaggingClassifier(n_estimators=estimator,
                            random_state=seed)

lg_clf = LGBMClassifier(n_estimators=estimator,
                        num_leaves=100,
                        verbosity=0,
                        random_state=seed,
                        n_jobs=n_jobs)

rf_clf = RandomForestClassifier(n_estimators=estimator,
                                min_samples_leaf=1,
                                verbose=0,
                                random_state=seed,
                                n_jobs=n_jobs,max_features = 15, max_depth = 35,)

et_clf = ExtraTreesClassifier(n_estimators=estimator,
                                min_samples_leaf=1,
                                verbose=0,
                                random_state=seed,
                                n_jobs=n_jobs)


# In[ ]:


sclf = StackingCVClassifier(classifiers=[bag_clf, lg_clf, rf_clf],
                            meta_classifier=et_clf,
                            random_state=seed,cv=3,verbose=2,n_jobs=n_jobs)


# In[ ]:


sclf.fit(X,y)


# In[ ]:


predict_SCLF = sclf.predict(X_test)
output = pd.DataFrame({'Id': ID,
                       'Cover_Type': predict_SCLF})
output.to_csv('submission_predict_SCLF.csv', index=False)
from IPython.display import FileLink
FileLink(r'submission_predict_SCLF.csv')


# In[ ]:




