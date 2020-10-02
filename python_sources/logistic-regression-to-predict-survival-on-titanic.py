#!/usr/bin/env python
# coding: utf-8

# # Data import

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette = 'deep')
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.dtypes


# # Univariate analysis

# In[ ]:


train.PassengerId.describe()


# In[ ]:


train.Survived.describe()


# In[ ]:


train.Survived.value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(x=train.Survived.value_counts().index, y=train.Survived.value_counts().values)
plt.title("Number of passengers deceased vs survived", fontdict = {'fontsize':20}, pad = 30.0)
labels = ["Deceased","Survived"]
plt.xticks(np.arange(2), labels)
plt.xlabel("Deceased vs survived")
plt.ylabel("Number of passengers")


# In[ ]:


train.Pclass.describe()


# In[ ]:


train.Pclass.value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(x=train.Pclass.value_counts().index, y=train.Pclass.value_counts().values)
plt.title("Number of passengers per class", fontdict = {'fontsize':20}, pad = 30.0)
labels = ["Class 1","Class 2","Class 3"]
plt.xticks(np.arange(3), labels)
plt.xlabel("Passenger class")
plt.ylabel("Number of passengers")


# In[ ]:


train.Name.describe()


# In[ ]:


train.Sex.describe()


# In[ ]:


train.Sex.value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(x=train.Sex.value_counts().index, y=train.Sex.value_counts().values)
plt.title("Number of passengers per sex", fontdict = {'fontsize':20}, pad = 30.0)
labels = ["Male","Female"]
plt.xticks(np.arange(2), labels)
plt.xlabel("Sex")
plt.ylabel("Number of passengers")


# In[ ]:


train['Male'] = 0
for i in train.index:
    if train['Sex'].iloc[i] == 'male':
        train['Male'].iloc[i] = 1
    elif train['Sex'].iloc[i] == 'female':
        train['Male'].iloc[i] = 0


# In[ ]:


train['Male'].value_counts()


# In[ ]:


train.Age.describe()


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.distplot(train[train['Age'].notnull()].Age, hist = True)
plt.title("Age distribution amongst passengers", fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Age")
plt.ylabel("Number of passengers")


# In[ ]:


fig, ax = plt.subplots(figsize=(4,6))
sns.boxplot(train[train['Age'].notnull()].Age, orient = "v", width = .3)
plt.title("Boxplot of age amongst passengers", fontdict = {'fontsize':20}, pad = 30.0)
plt.ylabel("Age")


# In[ ]:


train.SibSp.describe()


# In[ ]:


train.SibSp.value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x = train.SibSp.value_counts().index, y = train.SibSp.value_counts().values)
plt.title("Passengers with Siblings or Spouses on board", fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Number of siblings and/or spouse")
plt.ylabel("Number of passengers")


# In[ ]:


train.Parch.describe()


# In[ ]:


train.Parch.value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x = train.Parch.value_counts().index, y = train.Parch.value_counts().values)
plt.title("Passengers with Children or Parents on board", fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Number of children and/or parents")
plt.ylabel("Number of passengers")


# In[ ]:


train.Ticket.describe()


# In[ ]:


train.Fare.describe()


# In[ ]:


fig, ax = plt.subplots(figsize=(4,6))
sns.boxplot(train['Fare'], orient = "v", width = .3)
plt.title("Boxplot of fare amongst passengers", fontdict = {'fontsize':20}, pad = 30.0)
plt.ylabel("Fare")


# In[ ]:


fig, ax = plt.subplots(figsize=(4,6))
sns.boxplot(train[train['Fare'] <= 100].Fare, orient = "v", width = .3)
plt.title("Boxplot of fare amongst passengers who paid $100 or less", fontdict = {'fontsize':20}, pad = 30.0)
plt.ylabel("Fare")


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.distplot(train[train['Fare'] < 100].Fare, hist = True)
plt.title("Fare distribution amongst passengers who paid $100 or less", fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Fare")
plt.ylabel("Number of passengers")


# In[ ]:


train.Cabin.describe()


# In[ ]:


train['HasCabin'] = 0
for i in train.index:
    if isinstance(train['Cabin'].iloc[i], float):
        train['HasCabin'].iloc[i] = 1
    if isinstance(train['Cabin'].iloc[i], str):
        train['HasCabin'].iloc[i] = 0


# In[ ]:


train.HasCabin.value_counts()


# In[ ]:


train.Embarked.describe()


# In[ ]:


train.Embarked.value_counts()


# In[ ]:


ports = ["Southampton","Cherbourg","Queenstown"]

fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x = train.Embarked.value_counts().index, y = train.Embarked.value_counts().values)
plt.title("Number of passengers per port of embarkment", fontdict = {'fontsize':20}, pad = 30.0)
plt.xticks(np.arange(3), ports)
plt.xlabel("Port of embarkment")
plt.ylabel("Number of passengers")


# In[ ]:


train['Southampton'] = 0
train['Cherbourg'] = 0
train['Queenstown'] = 0

for i in train.index:
    if train.Embarked.iloc[i] == 'S':
        train.Southampton.iloc[i] = 1
    if train.Embarked.iloc[i] == 'C':
        train.Cherbourg.iloc[i] = 1
    if train.Embarked.iloc[i] == 'Q':
        train.Queenstown.iloc[i] = 1


# In[ ]:


train.Southampton.value_counts()


# In[ ]:


train.Cherbourg.value_counts()


# In[ ]:


train.Queenstown.value_counts()


# # Predicting Survival - Bivariate analysis

# In[ ]:


fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(train.corr(), cmap = 'Spectral', center = 0)
plt.title("Heatmap of correlations", fontdict = {'fontsize':20}, pad = 30.0)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.distplot(train[(train['Age'].notnull()) & (train['Survived'] == 0)].Age, label = 'Deceased')
sns.distplot(train[(train['Age'].notnull()) & (train['Survived'] == 1)].Age, label = 'Survived')
plt.title("Age distribution amongst deceased vs survived passengers", fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Age")
plt.ylabel("Number of passengers")
plt.legend()
plt.show


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.distplot(train[(train['Fare'].notnull()) & (train['Survived'] == 0)].Fare, label = 'Deceased')
sns.distplot(train[(train['Fare'].notnull()) & (train['Survived'] == 1)].Fare, label = 'Survived')
plt.title("Fare distribution amongst deceased vs survived passengers", fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Fare")
plt.ylabel("Number of passengers")
plt.legend()
plt.show


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.distplot(train[(train['Fare'].notnull()) & (train['Survived'] == 0)  & (train['Fare'] <= 100)].Fare, label = 'Deceased')
sns.distplot(train[(train['Fare'].notnull()) & (train['Survived'] == 1)  & (train['Fare'] <= 100)].Fare, label = 'Survived')
plt.title("Fare distribution amongst deceased vs survived passengers", fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Fare")
plt.ylabel("Number of passengers")
plt.legend()
plt.show


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.countplot(x = train.Parch, hue = train.Survived)
plt.title("Passengers with Children or Parents on board - Survived vs Deceased",
          fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Number of children and/or parents")
plt.ylabel("Number of passengers")


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.countplot(x = train.Pclass, hue = train.Survived)
plt.title("Survived vs Deceased Passengers per passenger class",
          fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Passenger class")
plt.ylabel("Number of passengers")


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.countplot(x = train.SibSp, hue = train.Survived)
plt.title("Passengers with Siblings or Spouses on board - Survived vs Deceased",
          fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Number of siblings and/or spouse")
plt.ylabel("Number of passengers")


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.countplot(x = train.Male, hue = train.Survived)
plt.title("Survived vs Deceased passengers by gender", fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Gender")
plt.xticks(np.arange(2), ("Female","Male"))
plt.ylabel("Number of passengers")


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.countplot(x = train.HasCabin, hue = train.Survived)
plt.title("Survived vs Deceased passengers by having a cabin", fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Has a cabin?")
plt.xticks(np.arange(2), ("No cabin","Cabin"))
plt.ylabel("Number of passengers")


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.countplot(x = train.Southampton, hue = train.Survived)
plt.title("Survived vs Deceased passengers by boarding in Southampton vs other port",
          fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Boarded at Southampton?")
plt.xticks(np.arange(2), ("No","Yes"))
plt.ylabel("Number of passengers")


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.countplot(x = train.Cherbourg, hue = train.Survived)
plt.title("Survived vs Deceased passengers by boarding in Cherbourg vs other port",
          fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Boarded at Cherbourg?")
plt.xticks(np.arange(2), ("No","Yes"))
plt.ylabel("Number of passengers")


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.countplot(x = train.Queenstown, hue = train.Survived)
plt.title("Survived vs Deceased passengers by boarding in Queenstown vs other port",
          fontdict = {'fontsize':20}, pad = 30.0)
plt.xlabel("Boarded at Queenstown?")
plt.xticks(np.arange(2), ("No","Yes"))
plt.ylabel("Number of passengers")


# # Predicting survival on Titanic - model

# In[ ]:


logregdata = train[['Survived','Age','Pclass','HasCabin','Male','Fare']]
logregdata = logregdata.dropna().reset_index()


# In[ ]:


target = logregdata['Survived']
feat = logregdata[['Age','Pclass','HasCabin','Male','Fare']]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(feat, target)


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg.score(X_train, y_train)


# In[ ]:


logreg.score(X_test, y_test)


# In[ ]:


logreg.predict(X_test)


# In[ ]:


result = pd.DataFrame(logreg.predict(feat), columns=['Predicted'])
result['Actual'] = target.reset_index(drop=True)


# In[ ]:


confusion_matrix(result['Actual'], result['Predicted'])


# In[ ]:


sns.heatmap(confusion_matrix(result['Actual'], result['Predicted']), annot=True, cmap = 'Blues', fmt='g')


# # Logistic Regression on Age, Fare, Sex, Cabin and Pclass

# In[ ]:


test['Male'] = 0
for i in test.index:
    if test['Sex'].iloc[i] == 'male':
        test['Male'].iloc[i] = 1
    elif test['Sex'].iloc[i] == 'female':
        test['Male'].iloc[i] = 0


# In[ ]:


test['HasCabin'] = 0
for i in test.index:
    if isinstance(test['Cabin'].iloc[i], float):
        test['HasCabin'].iloc[i] = 1
    if isinstance(test['Cabin'].iloc[i], str):
        test['HasCabin'].iloc[i] = 0


# In[ ]:


feat_test = test[['PassengerId','Age','Male','HasCabin','Fare','Pclass']]


# In[ ]:


feat_test[feat_test.isnull().any(axis=1)]


# In[ ]:


train['Age'].append(test['Age']).mean()


# In[ ]:


feat_test['Age'] = feat_test['Age'].fillna(30)


# In[ ]:


feat_test[feat_test.isnull().any(axis=1)]


# In[ ]:


train['Fare'].append(test['Fare']).mean()


# In[ ]:


feat_test['Fare'] = feat_test['Fare'].fillna(34.28)


# In[ ]:


feat_test[feat_test.isnull().any(axis=1)]


# In[ ]:


np.set_printoptions(suppress=True)
results = pd.Series(logreg.predict(feat_test[['Age','Male','HasCabin','Fare','Pclass']]))


# In[ ]:


feat_test['Survived'] = results


# In[ ]:


output = feat_test[['PassengerId','Survived']].astype('int64')


# In[ ]:


output.to_csv('LogReg_results.csv',index=False)


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


dt_train_feat = train[['Pclass','Age','SibSp','Parch','Fare','Male','HasCabin']]


# In[ ]:


dt_train_feat['Age'] = dt_train_feat['Age'].fillna(30)
dt_train_feat['Fare'] = dt_train_feat['Fare'].fillna(34.28)


# In[ ]:


dt_train_feat[dt_train_feat.isnull().any(axis=1)]


# In[ ]:


dt_train_target = train['Survived']


# In[ ]:


dtc = DecisionTreeClassifier()
dtc.fit(dt_train_feat, dt_train_target)


# In[ ]:


dtc.score(dt_train_feat, dt_train_target)


# In[ ]:


dt_test_feat = test[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Male','HasCabin']]


# In[ ]:


dt_test_feat['Age'] = dt_test_feat['Age'].fillna(30)
dt_test_feat['Fare'] = dt_test_feat['Fare'].fillna(34.28)


# In[ ]:


dt_test_feat[dt_test_feat.isnull().any(axis=1)]


# In[ ]:


dt_results = pd.Series(dtc.predict(dt_test_feat[['Pclass','Age','SibSp','Parch','Fare','Male','HasCabin']]))


# In[ ]:


dt_test_feat['Survived'] = dt_results


# In[ ]:


output = dt_test_feat[['PassengerId','Survived']].astype('int64')


# In[ ]:


output.to_csv('DecTree_results.csv',index=False)

