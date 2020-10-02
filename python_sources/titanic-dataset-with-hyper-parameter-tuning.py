#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns


# In[ ]:


df_Train=pd.read_csv('../input/titanic/train.csv')
df_Test=pd.read_csv('../input/titanic/test.csv')


# In[ ]:


df_Train.head()


# In[ ]:


df_Test.head()


# In[ ]:


df_Train.isnull().sum()


# In[ ]:


sns.heatmap(df_Train.isnull(),yticklabels=False,cbar=False)


# In[ ]:


df_Train.drop(['Cabin'],axis=1,inplace=True)
df_Train.drop(['Name'],axis=1,inplace=True)
df_Train.drop(['Ticket'],axis=1,inplace=True)
df_Train['Age']=df_Train['Age'].fillna(df_Train['Age'].mode()[0])
sns.heatmap(df_Train.isnull(),yticklabels=False,cbar=False)


# In[ ]:


sns.heatmap(df_Test.isnull(),yticklabels=False,cbar=False)


# In[ ]:


df_Test.drop(['Cabin'],axis=1,inplace=True)
df_Test.drop(['Name'],axis=1,inplace=True)
df_Test.drop(['Ticket'],axis=1,inplace=True)
df_Test['Age']=df_Test['Age'].fillna(df_Test['Age'].mode()[0])
df_Test['Fare']=df_Test['Fare'].fillna(df_Test['Fare'].mode()[0])
sns.heatmap(df_Test.isnull(),yticklabels=False,cbar=False)


# In[ ]:


df_Test.isnull().sum()


# In[ ]:


df_Train.describe()


# In[ ]:


df_Train.head()


# In[ ]:


S_Dummy=pd.get_dummies(df_Train['Sex'],drop_first=True)
Embarked_Dummy=pd.get_dummies(df_Train['Embarked'],drop_first=True)


# In[ ]:


df_Train=pd.concat([df_Train,S_Dummy,Embarked_Dummy],axis=1)
df_Train


# In[ ]:


df_Train.drop(['Sex','Embarked'],axis=1,inplace=True)
df_Train.head()


# In[ ]:


#Using Random forest for training data itself
from sklearn.model_selection import train_test_split
X=df_Train[['PassengerId','Pclass','Age','SibSp','Parch','Fare','male','Q','S']]
y=df_Train[['Survived']]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)


# In[ ]:


pred = rf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(accuracy_score(y_test, pred))


# In[ ]:


## Trying to tune the hyperparameters (the parameters which can be decided by us and not by the model)

## Using Cross Validation for the same
from sklearn.model_selection import RandomizedSearchCV

# Define the parameter sets
# int(x) for x in ... converts the array into list. 

n_estimators = [int(x) for x in np.arange(50, 65, 5)]
max_features = ['auto','sqrt']
max_depth = [int(x) for x in np.arange(1,20,4)]
max_depth.append(None)
min_samples_leaf = [1,2,3,4,5]
min_samples_split = [3,4,5,6,7,8,9]
bootstrap = [True,False]


param_grid = {'n_estimators' : n_estimators,
              'max_features' : max_features,
              'max_depth' : max_depth,
              'min_samples_leaf' : min_samples_leaf,
              'min_samples_split' : min_samples_split,
              'bootstrap' : bootstrap}


rf1 = RandomForestClassifier()
rf1_random = RandomizedSearchCV(rf1, param_grid, n_iter = 110, cv = 6, verbose =2 , random_state = 130, n_jobs = -1)


# In[ ]:


rf1_random.fit(X_train, y_train)


# In[ ]:


print(rf1_random.best_score_)
rf1_random.best_params_


# In[ ]:


rf1_random.best_estimator_


# In[ ]:


pred2 = rf1_random.best_estimator_.predict(X_test)
print(accuracy_score(y_test, pred2))


# In[ ]:


Sx_Dummy=pd.get_dummies(df_Test['Sex'],drop_first=True)
Embark_Dummy=pd.get_dummies(df_Test['Embarked'],drop_first=True)
df_test=pd.concat([df_Test,Sx_Dummy,Embark_Dummy],axis=1)
df_test


# In[ ]:


df_test.drop(['Sex','Embarked'],axis=1,inplace=True)
df_test.head()


# In[ ]:


X=df_Train[['PassengerId','Pclass','Age','SibSp','Parch','Fare','male','Q','S']]
y=df_Train[['Survived']]


# In[ ]:


## Fit the model with the best set of parameters
rf_fin = rf1_random.best_estimator_
rf_fin.fit(X,y)


# In[ ]:


prediction=rf_fin.predict(df_test)
prediction


# In[ ]:


output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': prediction})
output.to_csv('my_submission14.csv', index=False)


# In[ ]:




