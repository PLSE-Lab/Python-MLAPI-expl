#!/usr/bin/env python
# coding: utf-8

# In[181]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() # setting seaborn default for plots


import os
print(os.listdir("../input"))


# In[182]:


tr = pd.read_csv('../input/train.csv')
te = pd.read_csv('../input/test.csv')


# In[183]:


tr.head()


# In[184]:


tr.drop(columns="Name",inplace=True)
tr.drop(columns="Ticket",inplace=True)
tr.drop(columns="PassengerId",inplace=True)
te.drop(columns="Name",inplace=True)
te.drop(columns="Ticket",inplace=True)
te_pass_id = te['PassengerId']
te.drop(columns="PassengerId",inplace=True)
tr.head()


# In[185]:


tr['familyMembers'] = tr['SibSp']+tr['Parch']

tr.drop(columns="SibSp",inplace=True)
tr.drop(columns="Parch",inplace=True)

te['familyMembers'] = te['SibSp']+te['Parch']

te.drop(columns="SibSp",inplace=True)
te.drop(columns="Parch",inplace=True)


# In[186]:


tr.info()


# In[187]:


tr.drop(columns="Cabin",inplace=True)
te.drop(columns="Cabin",inplace=True)


# In[188]:


tr['Age'].mean()


# In[189]:


tr['Age'] = tr['Age'].fillna(tr['Age'].mean())
te['Age'] = te['Age'].fillna(te['Age'].mean())
print(tr.head())
print(tr.info())
print(tr.dtypes)
print(tr.describe(include=['O']))


# In[190]:


tr['Sex'] = tr['Sex'].map({'male':0,'female':1})
te['Sex'] = te['Sex'].map({'male':0,'female':1})
tr.head()


# In[191]:


tr['Embarked'].value_counts()


# In[192]:


tr.groupby('Embarked').Survived.value_counts()


# In[193]:


tr['Embarked'] = tr['Embarked'].fillna('S')
te['Embarked'] = te['Embarked'].fillna('S')
tr.head()


# In[194]:


print(tr.head())
print('-'*50)
print(tr.info())
print('-'*50)
print(tr.dtypes)
print('-'*50)
print(tr.describe(include=['O']))


# In[195]:


tr['Embarked'] = tr['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
te['Embarked'] = te['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[196]:


print(tr.head())
print('-'*50)
print(tr.info())
print('-'*50)
print(tr.dtypes)


# In[197]:


print(te.head())
print('-'*50)
print(te.info())
print('-'*50)
print(te.dtypes)


# In[198]:


te['Fare'] = te['Fare'].fillna(te['Fare'].mean())


# In[199]:


X_train = tr.drop('Survived', axis=1)
y_train = tr['Survived']
X_test = te.copy()
X_train.shape, y_train.shape, X_test.shape


# In[200]:


# Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier


# In[201]:


clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)
print (str(acc_log_reg) + ' percent')


# In[202]:


clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train) * 100, 2)
print (acc_svc)


# In[203]:


clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)
print (acc_linear_svc)


# In[204]:


clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) * 100, 2)
print (acc_knn)


# In[205]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print (acc_decision_tree)


# In[206]:


clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print (acc_random_forest)


# In[207]:


clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, y_train) * 100, 2)
print (acc_gnb)


# In[208]:


clf = Perceptron(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_perceptron = clf.predict(X_test)
acc_perceptron = round(clf.score(X_train, y_train) * 100, 2)
print (acc_perceptron)


# In[209]:


clf = SGDClassifier(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_sgd = clf.predict(X_test)
acc_sgd = round(clf.score(X_train, y_train) * 100, 2)
print (acc_sgd)


# In[210]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
              'Perceptron', 'Stochastic Gradient Decent'],
    
    'Score': [acc_log_reg, acc_svc, acc_linear_svc, 
              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, 
              acc_perceptron, acc_sgd]
    })

models.sort_values(by='Score', ascending=False)


# In[211]:


te.head()


# In[212]:


submission = pd.DataFrame({
        "PassengerId": te_pass_id,
        "Survived": y_pred_random_forest
    })


# In[213]:


submission.to_csv('submission.csv', index=False)


# <a href="submission.csv"> Download File </a>
# 

# In[ ]:




