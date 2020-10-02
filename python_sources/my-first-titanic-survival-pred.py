#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()


# In[ ]:


train.dtypes


# In[ ]:


train.describe()


# In[ ]:


train.shape


# # Data Visualization

# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train)
plt.xlabel('Ticket Class')
plt.ylabel('Survived')
plt.show()


# In[ ]:


sns.barplot(x='Sex', y='Survived', data=train)
plt.xlabel('Sex')
plt.ylabel('Survived')
plt.show()


# In[ ]:


sns.barplot(x='SibSp', y='Survived', data=train)
plt.xlabel('Siblings or Spouses')
plt.ylabel('Survived')
plt.show()


# In[ ]:


sns.barplot(x='Parch', y='Survived', data=train)
plt.xlabel('Parent or Childerns')
plt.ylabel('Survived')
plt.show()


# In[ ]:


corr = train.corr()

sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# # Data Cleaning

# In[ ]:


def checkNull(df):
    total = df.isnull().sum()
    percent = (total / df.isnull().count()) * 100
    null_df = pd.concat([total, percent], keys=['Sum', 'Percent'], axis=1)
    return null_df


# In[ ]:


checkNull(train)


# In[ ]:


checkNull(test)


# In[ ]:


train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train['Cabin'].fillna(train['Cabin'].mode()[0], inplace=True)
train['Age'].fillna(0, inplace=True)
train['Title'] = train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
train['Title'].unique()


# In[ ]:


test['Age'].fillna(0, inplace=True)
test['Cabin'].fillna(test['Cabin'].mode()[0], inplace=True)
test.dropna(inplace=True)
test['Title'] = train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
test['Title'].unique()


# In[ ]:


checkNull(train)


# In[ ]:


checkNull(test)


# In[ ]:


def changeTitle(df):
    df['Title'] = df['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    df['Title'] = df['Title'].astype('int64')


# In[ ]:


changeTitle(train)
changeTitle(test)


# In[ ]:


sex_mapping = {'male':1, 'female':2}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)


# In[ ]:


age_bins = [-1, 0, 3, 14, 18, 30, 80]
age_labels = ['Unknown', 'Infant', 'Childern', 'Teenagers', 'Adult', 'Old']
train['AgeGroup'] = pd.cut(train['Age'], age_bins, labels=age_labels)
test['AgeGroup'] = pd.cut(test['Age'], age_bins, labels=age_labels)


# In[ ]:


age_mapping = {'Unknown': 0, 'Infant': 1, 'Childern': 2, 'Teenagers': 3, 'Adult': 4, 'Old': 5}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)


# In[ ]:


emb_mapping = {'S': 0, 'C': 1, 'Q': 2}
train['Embarked'] = train['Embarked'].map(emb_mapping)
test['Embarked'] = test['Embarked'].map(emb_mapping)


# In[ ]:


train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])


# In[ ]:


train.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)
test.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# # Prediction & Evaluation

# In[ ]:


x = train.drop('Survived', axis=1)
y = train['Survived']


# **1. KNN**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

ks = list(range(1,51))
k = 0
weight_options = ["uniform", "distance"]
param_grid = dict(n_neighbors = ks, weights = weight_options)

knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')
grid.fit(x_train,y_train)

knn = KNeighborsClassifier(grid.best_params_['n_neighbors'])


# In[ ]:


knn.fit(x_train,y_train)
yhat = knn.predict(x_test)
kscore = knn.score(x_test, y_test)
kscore


# **2. Gaussian Distribution**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train,y_train)
yhat = gaussian.predict(x_test)
gscore = gaussian.score(x_test,y_test)
gscore


# **3. Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train,y_train)
yhat = lr.predict(x_test)
lscore = lr.score(x_test, y_test)
lscore


# **4. Random Forest Classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier()
forest.fit(x_train,y_train)
yhat = forest.predict(x_test)
fscore = forest.score(x_test, y_test)
fscore


# In[ ]:


models = pd.DataFrame({
    'Model': ['KNN', 'Gaussian Distribution', 'Logistic Regression', 'Random Forest'],
    'Score': [kscore, gscore, lscore, fscore]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


yhat = forest.predict(test)


# In[ ]:


submission = pd.DataFrame({ 'PassengerId' : test['PassengerId'], 'Survived': yhat })
submission.to_csv('submission.csv', index=False)


# In[ ]:




