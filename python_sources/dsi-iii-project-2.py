#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.shape, df_test.shape


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


# check if passengerid unique
df_train.PassengerId.value_counts().sum(), df_test.PassengerId.value_counts().sum()


# In[ ]:


df_train.set_index('PassengerId', inplace=True), df_test.set_index('PassengerId', inplace=True)


# In[ ]:


# join train and test
data = pd.concat((df_train, df_test), sort=False)


# In[ ]:


# check dtypes and missing values
data.info()


# In[ ]:


# check null values
null_totals = data.drop('Survived', axis=1).isnull().sum().sort_values(ascending=False)
null_totals = null_totals[null_totals != 0]
plt.figure(figsize=(15,8))
sns.barplot(null_totals.index, null_totals)
plt.xticks(rotation=90)
plt.gca().set_ylabel('Number of NaN values');


# In[ ]:


# relatively few missing values to fill, lets start with Fare
data[data.Fare.isnull()]


# In[ ]:


data.Fare.fillna(data.Fare.dropna().mode()[0], inplace=True)
data.Fare.isnull().sum()


# In[ ]:


data[data.Embarked.isnull()]


# In[ ]:


data.Embarked.fillna(data.Embarked.mode()[0], inplace=True)
data.Embarked.isnull().sum()


# In[ ]:


data[data.Age.isnull()].head()


# In[ ]:


# lets fill missing ages with the median grouped by sex and class
data.Age = data.groupby(['Pclass', 'Sex']).Age.apply(lambda x: x.fillna(x.median()))
data.Age.isnull().sum()


# In[ ]:


data.Cabin.isnull().sum()


# In[ ]:


# lets create a feature for passengers with cabins then drop the cabins column
data['WithCabin'] = data.Cabin.apply(lambda x: 1 if type(x) == str else 0)
data.drop('Cabin', axis=1, inplace=True)


# In[ ]:


# it doesn't seem that ticket is a useful feature so we can drop it
data.drop('Ticket', axis=1, inplace=True)


# In[ ]:


# lets create a new column total family and drop sibsp and parch
data['TotalFamily'] = data.SibSp + data.Parch + 1
data.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# In[ ]:


# lets add an alone feature if no family members with the passenger
data['isAlone'] = data.TotalFamily.apply(lambda x: 1 if x == 1 else 0)


# In[ ]:


# create a title column using regex
regex = r', (.{1,13})\.'
data['Title'] = data.Name.apply(lambda x: re.findall(regex, x)[0])
data.Title.value_counts()


# In[ ]:


# fix titles
data.Title = data.Title.apply(lambda x: 'Miss' if x in ['Mlle', 'Ms'] else 'Mrs' if x == 'Mme' else x)
popular_titles = ['Mr', 'Miss', 'Mrs', 'Master']
data.Title = data.Title.apply(lambda x: x if x in popular_titles else 'Other')
data.Title.value_counts()


# In[ ]:


data.head()


# In[ ]:


data['NameLength'] = data.Name.apply(lambda x: len(x))
data.drop('Name', axis=1, inplace=True)


# In[ ]:


# dummify sex value
data.Sex = data.Sex.map({'male':1, 'female':0})
data.Sex.value_counts()


# In[ ]:


def dummify_column(data, column):
    return pd.concat((data, pd.get_dummies(data[column.name], prefix=column.name, prefix_sep='_is')), axis=1)


# In[ ]:


# dummify columns
data = dummify_column(data, data.Embarked)
data = dummify_column(data, data.Title)
data = dummify_column(data, data.Pclass)
data.drop('Embarked', axis=1, inplace=True)
data.drop('Title', axis=1, inplace=True)
data.drop('Pclass', axis=1, inplace=True)
data.head()


# In[ ]:


# lets check a couple of scatter plots
plt.figure(figsize=(9,5))
sns.regplot(data.Fare, data.Survived)
sns.regplot(data.Age, data.Survived)
plt.gca().set_xlabel('Age(Orange) & Fare(Blue)');


# It seems that the older you are the less likely you are to survive. And the higher the fare, the more likely someone will survive

# In[ ]:


# lets split age & fare into bins and dummify them 
pd.cut(data.Age, 5).value_counts()


# In[ ]:


pd.qcut(data.Fare, 4).value_counts()


# In[ ]:


data.Age = data.Age.apply(lambda x: 0 if x <= 16 else 1 if x <= 32 else 2 if x <= 48 else 3 if x <= 64 else 4)


# In[ ]:


data.Fare = data.Fare.apply(lambda x: 0 if x <= 7.9 else 1 if x <= 14.5 else 2 if x <= 31.3 else 3)


# In[ ]:


data.info()


# In[ ]:


plt.figure(figsize=(16,16))
sns.heatmap(data[data.Survived.notnull()].corr(), center=0, cmap='coolwarm', annot=True);


# Time to split data back into train and test and do modeling

# In[ ]:


y = data.Survived[data.Survived.notna()]
X = data.drop('Survived', axis=1)
X_train = X[data.Survived.notna()]
X_test = X[data.Survived.isnull()]
X_train.shape, X_test.shape, y.shape


# In[ ]:


# modeling imports for KNN, LogReg, RandomForest, adaboost, SVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.pipeline import Pipeline


# In[ ]:


# create scaled X
ss = StandardScaler()
X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.transform(X_test)


# In[ ]:


# creata a pipeline for each model to be used in gridsearchCV
cv = StratifiedKFold(n_splits=10, shuffle=True)
scaler = StandardScaler()
knn = KNeighborsClassifier()
knn_pipeline = Pipeline([('transformer', scaler), ('estimator', knn)])

knn_params = {'estimator__n_neighbors': [1,3,5,7,9,11,13,15,21], 'estimator__weights':['uniform', 'distance'],
             'estimator__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}

knn_grid = GridSearchCV(knn_pipeline, knn_params, n_jobs=-1, cv=cv, verbose=2)
knn_grid.fit(X_train, y);


# In[ ]:


knn_grid.best_score_, knn_grid.best_params_


# In[ ]:


cv = StratifiedKFold(n_splits=10, shuffle=True)
scaler = StandardScaler()
logreg = LogisticRegression()
log_pipeline = Pipeline([('transformer', scaler), ('estimator', logreg)])

log_params = {'estimator__penalty': ['l1', 'l2'],
              'estimator__C':[.00001, .0001, .001, .005, .01, .025, .05, .075, .1, .25, .5, .75, 1, 1.5, 2, 5]}

log_grid = GridSearchCV(log_pipeline, log_params, n_jobs=-1, cv=cv, verbose=2)
log_grid.fit(X_train, y);


# In[ ]:


log_grid.best_score_, log_grid.best_params_


# In[ ]:


cv = StratifiedKFold(n_splits=10, shuffle=True)
scaler = StandardScaler()
forest = RandomForestClassifier()
forest_pipeline = Pipeline([('transformer', scaler), ('estimator', forest)])

forest_params = {'estimator__n_estimators': [50, 150, 1000, 2000],
              'estimator__max_depth':[2, 3, 5, 7, 9, 11, 18],
                'estimator__max_features':[2, 3, 5, 7, 9, 11]}

forest_grid = GridSearchCV(forest_pipeline, forest_params, n_jobs=-1, cv=cv, verbose=2)
forest_grid.fit(X_train, y);


# In[ ]:


forest_grid.best_score_, forest_grid.best_params_


# In[ ]:


cv = StratifiedKFold(n_splits=10, shuffle=True)
scaler = StandardScaler()
ada = AdaBoostClassifier()
ada_pipeline = Pipeline([('transformer', scaler), ('estimator', ada)])

ada_params = {'estimator__base_estimator': [None, DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3)],
              'estimator__n_estimators': [150, 300, 1500],
              'estimator__learning_rate':[.01, .1, .5, 1, 2]}

ada_grid = GridSearchCV(ada_pipeline, ada_params, n_jobs=-1, cv=cv, verbose=2)
ada_grid.fit(X_train, y);


# In[ ]:


ada_grid.best_score_, ada_grid.best_params_


# In[ ]:


cv = StratifiedKFold(n_splits=10, shuffle=True)
scaler = StandardScaler()
svc = SVC()
svc_pipeline = Pipeline([('transformer', scaler), ('estimator', svc)])

svc_params = {'estimator__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
              'estimator__C':[.0001,.001,.005,.01,.05,.1,.25, .5, .75, 1,5,10,15,20]}


svc_grid = GridSearchCV(svc_pipeline, svc_params, n_jobs=-1, cv=cv, verbose=2)
svc_grid.fit(X_train, y);


# In[ ]:


svc_grid.best_score_, svc_grid.best_params_


# In[ ]:


# save each model
best_knn = knn_grid.best_estimator_
best_log = log_grid.best_estimator_
best_forest = forest_grid.best_estimator_
best_ada = ada_grid.best_estimator_
best_svc = svc_grid.best_estimator_


# In[ ]:


# we'll try stacking, lets create a df with predictions from each model as a feature
metadf = pd.DataFrame({'knn':best_knn.predict(X), 'logreg':best_log.predict(X), 'random_forest':best_forest.predict(X),
              'adaboost':best_ada.predict(X), 'svc':best_svc.predict(X), 'survived':y}, index=X.index)


# In[ ]:


# lets check model correlation
plt.figure(figsize=(16,16))
sns.heatmap(metadf[metadf.survived.notna()].corr(), center=0, cmap='coolwarm', annot=True);


# In[ ]:


metaX_train = metadf[metadf.survived.notna()].drop('survived', axis=1)
metaX_test = metadf[metadf.survived.isnull()].drop('survived', axis=1)
metaX_train.shape, metaX_test.shape


# In[ ]:


# lets try bagging as a metamodel
cv = StratifiedKFold(n_splits=10, shuffle=True)

bag_params = {'n_estimators': [5000], 'base_estimator':[None, KNeighborsClassifier(n_neighbors=7), SVC(), SVC(kernel='linear'),
                                                        LogisticRegression(), LogisticRegression('l1')]}

bag_grid = GridSearchCV(BaggingClassifier(), bag_params, n_jobs=-1, cv=cv, verbose=2)
bag_grid.fit(metaX_train, y);


# In[ ]:


bag_grid.best_score_, bag_grid.best_params_


# In[ ]:


best_bag = bag_grid.best_estimator_


# In[ ]:


# lets export our results and score it on kaggle
knn_scoredf = pd.DataFrame({'PassengerId':X_test.index, 'Survived':best_knn.predict(X_test).astype(int)})
logreg_scoredf = pd.DataFrame({'PassengerId':X_test.index, 'Survived':best_log.predict(X_test).astype(int)})
forest_scoredf = pd.DataFrame({'PassengerId':X_test.index, 'Survived':best_forest.predict(X_test).astype(int)})
ada_scoredf = pd.DataFrame({'PassengerId':X_test.index, 'Survived':best_ada.predict(X_test).astype(int)})
svc_scoredf = pd.DataFrame({'PassengerId':X_test.index, 'Survived':best_svc.predict(X_test).astype(int)})
stack_bag_scoredf = pd.DataFrame({'PassengerId':X_test.index, 'Survived':best_bag.predict(metaX_test).astype(int)})


# In[ ]:


knn_scoredf.to_csv('knn_score.csv', index=False)
logreg_scoredf.to_csv('log_score.csv', index=False)
forest_scoredf.to_csv('forest_score.csv', index=False)
ada_scoredf.to_csv('ada_score.csv', index=False)
svc_scoredf.to_csv('svc_score.csv', index=False)
stack_bag_scoredf.to_csv('stackedbag_score.csv', index=False)

