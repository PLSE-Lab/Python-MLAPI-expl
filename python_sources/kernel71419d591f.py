#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import missingno as missing


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[6]:


train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')


# In[ ]:


train.head(10)


# In[ ]:


train.columns


# In[ ]:


train = train.reindex_axis(['PassengerId',  'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked','Survived'], axis =1 )


# In[ ]:


train.head(10)


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


missing.matrix(train)


# In[ ]:


x_train = train.drop(['PassengerId', 'Cabin','Name','Ticket', 'Survived'], axis=1)


# In[ ]:


x_train.head()


# In[ ]:


y_train = train.Survived


# In[ ]:


y_train.value_counts()


# In[ ]:


test.head()


# In[ ]:


x_test= test.drop(['PassengerId', 'Cabin','Name','Ticket'], axis=1)


# In[ ]:


x_test.head()


# In[ ]:


PassengerId = test.PassengerId


# In[ ]:


PassengerId.head()


# In[ ]:


x_train.isnull().sum()


# In[ ]:


x_train.Age.fillna(value= x_train.Age.median(), inplace= True)


# In[ ]:


x_train.Age.isnull().sum()


# In[ ]:


x_train.Embarked.fillna(value= x_train.Embarked.value_counts().argmax(),inplace= True )


# In[ ]:


x_train.Embarked.isnull().sum()


# In[ ]:


missing.matrix(x_train,figsize=(8,6))


# In[ ]:


missing.matrix(x_test)


# In[ ]:


x_test.isnull().sum()


# In[ ]:


x_test.Age.fillna(value= x_train.Age.median(), inplace= True)


# In[ ]:


x_test.Age.isnull().any()


# In[ ]:


x_test.Fare.head()


# In[ ]:


x_test.Fare.fillna(value = x_test.Fare.value_counts().argmax(), inplace=True)


# In[ ]:


x_test.Fare.isnull().any()


# In[ ]:


missing.matrix(x_test, figsize=(6,3))


# In[ ]:


sns.countplot(x_train.Sex)


# In[ ]:


x_train.head(10)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le = LabelEncoder()


# In[ ]:


x_train.Sex =  le.fit_transform(x_train.Sex)


# In[ ]:


x_train.head()


# In[ ]:


x_train = pd.get_dummies(x_train, columns=['Pclass', 'Embarked'], drop_first=True)


# In[ ]:


x_train.head()


# In[ ]:


x_test = pd.get_dummies(x_test, columns=['Pclass', 'Embarked'], drop_first=True)


# In[ ]:


x_test.Sex =  le.fit_transform(x_test.Sex)


# In[ ]:


x_test.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


sc  = StandardScaler()


# In[ ]:


x_train = sc.fit_transform(x_train)


# In[ ]:


x_test = sc.transform(x_test)


# In[ ]:


x_test


# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


tree = DecisionTreeClassifier(random_state=0)


# In[ ]:


tree.fit(x_train,y_train)


# In[ ]:


tree.score(x_train,y_train)


# In[ ]:


svm = SVC(random_state=0)


# In[ ]:


svm.fit(x_train,y_train)


# In[ ]:


svm.score(x_train,y_train)


# In[ ]:


logistic = LogisticRegression(random_state=0)


# In[ ]:


logistic.fit(x_train,y_train)


# In[ ]:


logistic.score(x_train,y_train)


# In[ ]:


naieve = GaussianNB()


# In[ ]:


naieve.fit(x_train,y_train)


# In[ ]:


naieve.score(x_train,y_train)


# In[ ]:


forest = RandomForestClassifier(n_estimators=100 , random_state=0)


# In[ ]:


forest.fit(x_train,y_train)


# In[ ]:


forest.score(x_train,y_train)


# In[ ]:


params = {'n_estimators':[10, 50,100,150,200,300],'max_depth':[2,3,5,7] , 'verbose':[0,1], 'min_samples_leaf':[1,2,3]}


# In[ ]:


clf_forst = GridSearchCV(estimator= forest, cv = 5 ,param_grid= params)


# In[ ]:


clf_forst.fit(x_train,y_train)


# In[ ]:


clf_forst.best_score_


# In[ ]:


clf_forst.best_params_


# In[ ]:


tree_params = {'criterion':['gini','entropy'], 'max_depth':[2,4,5] }


# In[ ]:


clf_tree = GridSearchCV(estimator= tree, param_grid= tree_params, cv= 5, n_jobs=-1)


# In[ ]:


clf_tree.fit(x_train,y_train)


# In[ ]:


clf_tree.best_score_


# In[ ]:


svm_params = {'C':[1,5,50,100],'degree':[1,3,5,7], 'kernel':['rbf','linear','sigmoid','poly']}


# In[ ]:


clf_svm = GridSearchCV(estimator=svm, cv=5, param_grid= svm_params)


# In[ ]:


clf_svm.fit(x_train,y_train)


# In[ ]:


clf_svm.best_score_


# In[ ]:


x_train.head()


# In[ ]:


clf_forst.best_params_


# In[ ]:


rf = RandomForestClassifier(n_estimators= 200, max_depth=5, random_state=0)


# In[ ]:


rf.fit(x_train,y_train)


# In[ ]:


predict = rf.predict(x_test)


# In[ ]:


predict


# In[ ]:


submit = pd.DataFrame({'PassengerId':PassengerId, 'Survived':predict})


# In[ ]:


submit.to_csv('submition.csv', index=False)


# In[ ]:


submit.head()


# In[ ]:




