#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import seaborn as sns
import random as rnd


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
combine=[train,test]


# In[ ]:


train.describe()


# In[ ]:


train.columns


# In[ ]:


train.head()


# In[ ]:


train.describe(include=['O'])


# In[ ]:


train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by=['Survived'], ascending=True)


# In[ ]:


train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by=['Survived'], ascending=True)


# In[ ]:


train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by=['Survived'], ascending=True)


# In[ ]:


train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by=['Survived'], ascending=True)


# In[ ]:


train['family']=train['SibSp']+train['Parch']+1
train[['family','Survived']].groupby(['family'],as_index=False).mean().sort_values(by=['Survived'], ascending=True)


# In[ ]:


test['family']=test['SibSp']+test['Parch']+1
test.head()


# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[ ]:


grid=sns.FacetGrid(train, col='Survived', row='Pclass', height=2, aspect=1.5)
grid.map(plt.hist,'Age')


# In[ ]:


grid = sns.FacetGrid(train, row='Embarked', col='Survived', height=2, aspect=1.5)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# In[ ]:


train=train.drop(['Ticket','Cabin'], axis=1)
test=test.drop(['Ticket','Cabin'], axis=1)


# In[ ]:


combine=[train,test]
print(train.shape)
print(test.shape)


# In[ ]:


for dataset in combine:
    dataset['Title']=dataset.Name.str.extract(' ([A-Za-z]+)\.')
pd.crosstab(train['Title'], train['Sex'])    


# In[ ]:


for dataset in combine:
    dataset['Title']=dataset['Title'].replace(['Capt','Col','Countess','Don','Jonkheer','Lady','Major','Rev','Sir','Dr'],'Rare')
    dataset['Title']=dataset['Title'].replace('Mlle','Miss')
    dataset['Title']=dataset['Title'].replace('Mme','Miss')
    dataset['Title']=dataset['Title'].replace('Ms','Miss')

    


# In[ ]:


pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()


# In[ ]:


train=train.drop(['Name','PassengerId'], axis=1)
test=test.drop(['Name'], axis=1)


# In[ ]:


train["Sex"] = train["Sex"].map({"male": 0, "female":1})
test["Sex"] = test["Sex"].map({"male": 0, "female":1})
    
train['Sex'].head()


# In[ ]:


train.head()


# In[ ]:


train['Embarked'].unique()


# In[ ]:


title_mapping = {"S": 1, "C": 2, "Q": 3, np.NaN: 4}

train['Embarked'] = train['Embarked'].map(title_mapping)
test['Embarked'] = test['Embarked'].map(title_mapping)

train.head()


# In[ ]:


train.info()


# In[ ]:


for i in range(0,2):
    for j in range(0,3):
        #print(i,j+1)
        temp_dataset=train[(train['Sex']==i) &  (train['Pclass']==j+1)]['Age'].dropna()
        #print(temp_dataset)
        #print(str(temp_dataset.median())+"  "+str(i)+"  "+str(j+1))
        train.loc[(train.Age.isnull()) & (train.Sex==i) & (train.Pclass==j+1),'Age']=int(temp_dataset.median())
        
for i in range(0,2):
    for j in range(0,3):
        #print(i,j+1)
        temp_dataset=test[(test['Sex']==i) &  (test['Pclass']==j+1)]['Age'].dropna()
        #print(temp_dataset)
        #print(str(temp_dataset.median())+"  "+str(i)+"  "+str(j+1))
        test.loc[(test.Age.isnull()) & (test.Sex==i) & (test.Pclass==j+1),'Age']=int(temp_dataset.median())        


# In[ ]:


train.head(15)


# In[ ]:


train['Age'] = pd.cut(train['Age'], bins=[0, 12, 50, 200], labels=['Child','Adult','Elder'])
test['Age'] = pd.cut(test['Age'], bins=[0, 12, 50, 200], labels=['Child','Adult','Elder'])
train['Age'].head()


# In[ ]:


title_mapping = {'Child': 1, 'Adult': 2, 'Elder': 3}

train['Age'] = train['Age'].map(title_mapping).astype(int)
test['Age'] = test['Age'].map(title_mapping).astype(int)
    

train.head()


# In[ ]:



train=train.drop(['SibSp','Parch'], axis=1)

test=test.drop(['SibSp','Parch'], axis=1)


# In[ ]:


train.head()


# In[ ]:


train.info()
test.info()


# In[ ]:


test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)


# In[ ]:


test.info()


# In[ ]:


train['Fare']=(train['Fare']- train['Fare'].mean())/np.std(train['Fare'])
train.head()


# In[ ]:


test['Fare']=(test['Fare']- test['Fare'].mean())/np.std(test['Fare'])
test.head()


# # Predictive Models
# ****

# In[ ]:


X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# **1.Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# **2.Support Vector Machines**

# In[ ]:


from sklearn.svm import SVC, LinearSVC
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# )>**3.KNN**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# > **Selecting the Random forest classifier**

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:




