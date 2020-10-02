#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_raw=pd.read_csv('../input/mushrooms.csv')


# In[ ]:


print(data_raw.shape)
data_raw.head()


# In[ ]:


le=preprocessing.LabelEncoder()
for column in data_raw.columns:
    data_raw[column]=le.fit_transform(pd.Series(data_raw[column]))


# In[ ]:


#Shuffling the dataset
np.random.shuffle(data_raw.as_matrix())


# In[ ]:


data_raw.head()


# In[ ]:


X=data_raw.iloc[:,1:]
y=data_raw['class'].values


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


enc=OneHotEncoder()
X=enc.fit_transform(X).toarray()


# In[ ]:


X.shape


# > DATA EXPLORATION

# In[ ]:


#Count of each class
print(pd.Series(y).value_counts())


# *Since the classes are not skewed,so we may choose to use accuracy as our evaluation metric*

# In[ ]:


#Univariate Analysis
data_raw['cap-shape'].value_counts().sort_index().plot.bar()
plt.xlabel('Cap-Shape')


# In[ ]:


data_raw['cap-surface'].value_counts().sort_index().plot.bar()
plt.xlabel('Cap-Surface')


# In[ ]:


data_raw['cap-color'].value_counts().sort_index().plot.bar()
plt.xlabel('Cap-Color')


# In[ ]:


data_raw['bruises'].value_counts().sort_index().plot.bar()
plt.xlabel('bruises')


# In[ ]:


data_raw['odor'].value_counts().sort_index().plot.bar()
plt.xlabel('Odour')


# In[ ]:


data_raw['gill-attachment'].value_counts().sort_index().plot.bar()
plt.xlabel('Gill-Attachment')


# In[ ]:


data_raw['gill-spacing'].value_counts().sort_index().plot.bar()
plt.xlabel('Gill-Spacing')


# In[ ]:


data_raw['gill-size'].value_counts().sort_index().plot.bar()
plt.xlabel('Gill-Size')


# In[ ]:


data_raw['gill-color'].value_counts().sort_index().plot.bar()
plt.xlabel('Gill-Color')


# In[ ]:


data_raw['stalk-shape'].value_counts().sort_index().plot.bar()
plt.xlabel('Stalk-Shape')


# In[ ]:


data_raw['stalk-root'].value_counts().sort_index().plot.bar()
plt.xlabel('Stalk-Root')


# > DATA CLEANING

# In[ ]:


#Checking for Duplicates
X_df=pd.DataFrame(X)
dup=X_df.duplicated()
g=X_df[dup==True].index
print(g)


# *So there are no duplicates*

# In[ ]:


#Checking for missing values
data_raw.isnull().sum()


# *There are no missing values in the dataset*

# > DEPLOYING DIFFERENT MODELS

# **LOGISTIC REGRESSION**

# In[ ]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('logreg',LogisticRegression()))
model=Pipeline(estimators)
seed=4
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
results=cross_val_score(model,X,y,cv=kfold,scoring='accuracy')
print(results.mean())


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=6)
logreg=LogisticRegression()
from sklearn.metrics import confusion_matrix
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
confusion_matrix(y_test,y_pred)


# **KNN**

# In[ ]:


#Kneighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
results=[]
for i in range(1,31):
    estimators=[]
    estimators.append(('standardize',StandardScaler()))
    estimators.append(('kneighbors',KNeighborsClassifier(n_neighbors=i)))
    model=Pipeline(estimators)
    seed=1
    kfold=StratifiedKFold(n_splits=5,random_state=seed)
    results.append(cross_val_score(model,X,y,cv=kfold,scoring='accuracy').mean())
    print('Result of k='+str(i)+"  : "+str(results[-1]))


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(np.arange(1,31),results)
plt.show()


# **BERNOULLI  NAIVE BAYES**

# In[ ]:


#Naive Bayes Classifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
estimators=[]
results=0
from sklearn.model_selection import StratifiedKFold
estimators.append(('standardize',StandardScaler()))
estimators.append(('naive_bayes',BernoulliNB()))
model=Pipeline(estimators)
seed=1
param_grid=[{'naive_bayes__alpha':[0.0001,0.001,0.003,0.01,0.03,0.1,0.3,1]}]
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
grid=GridSearchCV(model,param_grid,cv=kfold,scoring='accuracy')
grid.fit(X_train,y_train)
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
print(accuracy_score(y_test,grid.predict(X_test)))


# In[ ]:


model=BernoulliNB(alpha=0.00001)
model.fit(X_train,y_train)
y_test=model.predict(X_test)
accuracy_score(y_test,y_pred)


# **SVM**

# In[ ]:


#SVM
from sklearn.svm import SVC
from sklearn.model_selection import KFold
estimators=[]
results=0
estimators.append(('standardize',StandardScaler()))
estimators.append(('svm',SVC(kernel='linear')))
model=Pipeline(estimators)
seed=1
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
results=cross_val_score(model,X,y,cv=kfold,scoring='accuracy')
results.mean()


# In[ ]:


svm=SVC(kernel='linear')
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# **LINEAR DISCRIMINANT ANALYSIS**

# In[ ]:


#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
estimators=[]
results=0
estimators.append(('standardize',StandardScaler()))
estimators.append(('svm',LinearDiscriminantAnalysis()))
model=Pipeline(estimators)
seed=1
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
results=cross_val_score(model,X,y,cv=kfold,scoring='accuracy')
results.mean()


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
ld=LinearDiscriminantAnalysis()
ld.fit(X_train,y_train)
y_pred=ld.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# **DECISION TREE**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
estimators=[]
results=0
from sklearn.model_selection import StratifiedKFold
#estimators.append(('standardize',StandardScaler()))
estimators.append(('dec_tree',DecisionTreeClassifier(random_state=23)))
model=Pipeline(estimators)
seed=1
param_grid=[{'dec_tree__max_depth':[2,3,5,7,9],'dec_tree__min_samples_leaf':[1,2,3,4,5,6],'dec_tree__max_features':['auto','sqrt',None]}]
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
grid=GridSearchCV(model,param_grid,cv=kfold,scoring='accuracy')
grid.fit(X_train,y_train)
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
print(accuracy_score(y_test,grid.predict(X_test)))


# In[ ]:


model=DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy_score(y_test,y_pred)


# **RANDOM FOREST**

# In[ ]:


#Random forest with Grid Search Cv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
estimators=[]
results=0
from sklearn.model_selection import StratifiedKFold
#estimators.append(('standardize',StandardScaler()))
estimators.append(('ran_for',RandomForestClassifier(random_state=23)))
model=Pipeline(estimators)
seed=1
param_grid=[{'ran_for__max_depth':[2,3,5,7,9],'ran_for__min_samples_leaf':[1,2,3,4,5,6],'ran_for__max_features':['auto','sqrt',None]}]
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
grid=GridSearchCV(model,param_grid,cv=kfold,scoring='accuracy')
grid.fit(X_train,y_train)
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
print(accuracy_score(y_test,grid.predict(X_test)))


# **XGBOOST**

# In[ ]:


#Random forest with Grid Search Cv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
estimators=[]
results=0
from sklearn.model_selection import StratifiedKFold
#estimators.append(('standardize',StandardScaler()))
estimators.append(('xgboost',GradientBoostingClassifier(random_state=23)))
model=Pipeline(estimators)
seed=1
param_grid=[{'xgboost__max_depth':[2,3,5],'xgboost__min_samples_leaf':[1,2,3,4,5,6],'xgboost__max_features':['auto','sqrt',None]}]
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
grid=GridSearchCV(model,param_grid,cv=kfold,scoring='accuracy')
grid.fit(X_train,y_train)
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
print(accuracy_score(y_test,grid.predict(X_test)))


# *Thus, each model performs with an accuracy of almost 99 % . It is mainly because the data is too clean is too organised*

# In[ ]:




