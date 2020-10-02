#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train


# In[ ]:


train.head()


# In[ ]:


sns.countplot(x = 'Survived',hue= 'Pclass' , data = train)


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=train)


# In[ ]:


sns.distplot(train['Age'],bins =25)


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(train.isnull())


#                                            Dealing with Missing Values

# In[ ]:


train.isnull().sum()


# In[ ]:


train['Age'].value_counts()


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=train)


# Here we can observe that 'Pclass' that 
# Pclass 1 : Age 38
# Pclass 2 : Age 29
# Pclass 3 : Age 25

# In[ ]:


def impute(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
    
        elif Pclass==2:
            return 28
        else:
            return 25
    else:
        return Age


# In[ ]:


train['Age']=train[['Age','Pclass']].apply(impute,axis=1);


# In[ ]:


train['Age'].isnull().sum()


# In[ ]:


train['Cabin'].isnull().sum()


# As Shown above 'Cabin' column contains most of the NULL values so better is that, we should drop the 'Cabin' Column

# In[ ]:


train = train.drop('Cabin',axis=1)


# In[ ]:


train['Embarked'].isnull().sum()


# In[ ]:


train['Embarked'].value_counts()


# In[ ]:


train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])


# In[ ]:


sns.heatmap(train.isnull(),cmap='Accent')


#                             Dealing With Catagorical Feature

# In[ ]:


train['Sex'].value_counts()


# In[ ]:


train['Sex'].replace(to_replace=['male','female'],value=[0,1],inplace=True)


# In[ ]:


train['Embarked'].value_counts()


# In[ ]:


train['Embarked'].replace(to_replace=['S','C','Q'],value=[0,1,2],inplace=True)


# In[ ]:


train['Ticket'].value_counts()


# In[ ]:


train = train.drop('Ticket',axis=1)


# As shown above 'ticket' doesn't contribute to predict any contribution so,its better to drop the feature

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(train.corr())


#                                         Dealing with test dataset

# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(test.isnull())


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=test)


# In[ ]:


def impute1(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==43:
            return 
    
        elif Pclass==2:
            return 26
        else:
            return 23
    else:
        return Age


# In[ ]:


test['Age']=test[['Age','Pclass']].apply(impute1,axis=1);


# In[ ]:


test = test.drop('Cabin',axis=1)


# In[ ]:


test = test.drop('Ticket',axis=1)


# In[ ]:


test['Embarked'].value_counts()


# In[ ]:


test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])


# In[ ]:


test['Sex'].replace(to_replace=['male','female'],value=[0,1],inplace=True)


# In[ ]:


test['Embarked'].replace(to_replace=['S','C','Q'],value=[0,1,2],inplace=True)


# In[ ]:


test['Fare'].value_counts()


# In[ ]:


test['Fare'] = test['Fare'].fillna(test['Fare'].mean())


# In[ ]:


test.isnull().sum()


# In[ ]:


sns.heatmap(test.isnull())


#                                             ML Modelling

# In[ ]:


train.head()


# In[ ]:


X=train[['PassengerId','Pclass','Age','SibSp','Parch','Sex','Embarked']]


# In[ ]:


y=train['Survived']


# In[ ]:





# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lg = LogisticRegression()


# In[ ]:


pred = lg.fit(X,y)


# In[ ]:


pred.score(X,y)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


# In[ ]:


lg.fit(X_test,y_test)


# In[ ]:


y_pred = lg.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000],'max_iter' :[100, 500 , 1000]}]
grid_search = GridSearchCV(estimator = lg,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_score_


# In[ ]:


grid_search.best_params_


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


print(accuracy_score(y_test,y_pred))


# In[ ]:


test
test[0:5]


# In[ ]:


X2 = test[['PassengerId','Pclass','Age','SibSp','Parch','Sex','Embarked']]


# In[ ]:


y2_pred = lg.predict(X2)
print(y2_pred)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y2_pred
    })


# In[ ]:


submission.to_csv("titanic_submission.csv", index=False)


# In[ ]:


sub = pd.read_csv('titanic_submission.csv')


# In[ ]:


sub


# In[ ]:




