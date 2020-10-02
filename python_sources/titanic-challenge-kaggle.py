#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# ## Bar Chart for Categorical Features 

# In[ ]:


def barChar(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind = 'bar', stacked=True, figsize=(10,5))


# In[ ]:


barChar('Sex')


# women survived more than men

# In[ ]:


barChar('Pclass')


# In[ ]:


barChar('SibSp')


# In[ ]:


barChar('Parch')


# In[ ]:


train.isnull()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False, cbar=False,cmap='viridis')


# In[ ]:


#0----> not survived
# 1---> survived

sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)


# In[ ]:


#see the distribution of age
#it helps us to see the avg age of people in titanic
sns.distplot(train['Age'].dropna(), kde=False, color='blue',bins=40)


# In[ ]:


#countplot of sibling 
sns.countplot(x='SibSp', data=train)


# In[ ]:


#average fare of ticket
train['Fare'].hist(color='red',bins=40)


# # Feature Engineering

# In[ ]:


train_test_data = [train, test]

for dataset in train_test_data:
    dataset['Title']=dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)


# In[ ]:


train['Title'].value_counts()


# In[ ]:


test['Title'].value_counts()


# ### CHeck average age

# In[ ]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data = train, palette='winter')


# we fill the vacant space of age with average age

# In[ ]:


def inputAge(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
        
    else:
        return Age


# In[ ]:


def embarkNull(cols):
    emb = cols
    if pd.isnull(emb):
        return 'S'
    
    else:
        return emb


# In[ ]:


# now apply this function
train['Age'] = train[['Age','Pclass']].apply(inputAge,axis=1)


# In[ ]:


train['Embarked'] = train['Embarked'].apply(embarkNull)


# In[ ]:


#again check heat map
sns.heatmap(train.isnull(),yticklabels=False, cbar=False,cmap='viridis')


# In[ ]:


#again check heat map
sns.heatmap(test.isnull(),yticklabels=False, cbar=False,cmap='viridis')


# Our Age Column is all full

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:





# In[ ]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data = test, palette='winter')


# for test data set

# In[ ]:


def inputAge(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 41
        elif Pclass == 2:
            return 27
        else:
            return 24
        
    else:
        return Age


# In[ ]:


def embarkNull(cols):
    emb = cols
    if pd.isnull(emb):
        return 'S'
    
    else:
        return emb


# In[ ]:


# now apply this function
test['Age'] = test[['Age','Pclass']].apply(inputAge,axis=1)

test['Embarked'] = test['Embarked'].apply(embarkNull)


# In[ ]:


#again check heat map
sns.heatmap(test.isnull(),yticklabels=False, cbar=False,cmap='viridis')


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Drop the Cabin values

# In[ ]:


train.drop('Cabin', axis = 1, inplace = True)
test.drop('Cabin', axis = 1, inplace = True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False, cbar=False,cmap='viridis')


# In[ ]:


sns.heatmap(test.isnull(),yticklabels=False, cbar=False,cmap='viridis')


# In[ ]:


#fill missing fare
train['Fare'].fillna(train.groupby("Pclass")["Fare"].transform("median"),inplace=True)
test['Fare'].fillna(test.groupby("Pclass")["Fare"].transform("median"),inplace=True)


# In[ ]:


sns.heatmap(test.isnull(),yticklabels=False, cbar=False,cmap='viridis')


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
 
plt.show()


# In[ ]:


sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train)


# In[ ]:


plt.figure(figsize=(15,6))
sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)


# Convert Categorical Feature

# In[ ]:


for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


pd.crosstab(train['Title'],train['Sex'])


# In[ ]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',  	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# convert categroical title into numeric form

# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Age Feature

# In[ ]:


for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <=32), 'Age'] = 1
    dataset.loc[(dataset['Age'] >32) & (dataset['Age'] <= 48), 'Age'] =2
    dataset.loc[(dataset['Age'] >48) & (dataset['Age'] <= 64), 'Age'] =3
    dataset.loc[dataset['Age'] >64, 'Age'] =4


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train['FareBand'] = pd.qcut(train['Fare'], 4)
print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())


# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[ ]:


train.head()


# In[ ]:


train.Embarked.unique()


# In[ ]:


train.Embarked.value_counts()


# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


for dataset in train_test_data:
    #print(dataset.Embarked.unique())
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[ ]:


train.head()


# In[ ]:


for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


# In[ ]:


for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


# In[ ]:


train.head(2)


# In[ ]:


train.columns


# In[ ]:


test.columns


# # Feature Selection

# In[ ]:


features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'FamilySize']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'FareBand'], axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# # Make the Model

# In[ ]:


x_train = train.drop('Survived', axis = 1)
y_train = train['Survived']

x_test = test.drop('PassengerId', axis=1).copy()
x_train.shape, y_train.shape, x_test.shape


# In[ ]:





# In[ ]:


#importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier


# In[ ]:


x_train.info()


# ## CRoss Validation

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# ## KNN

# In[ ]:


clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, x_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accKNN = round(clf.score(x_train, y_train)*100, 2)
accKNN


# ## Random Forest

# In[ ]:


clf = RandomForestClassifier(n_estimators=15)
scoring = 'accuracy'
score = cross_val_score(clf, x_train, y_train, cv = k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


acc_RForest = round(np.mean(score)*100,2)


# ## Naive Bayes

# In[ ]:


clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, x_train, y_train, cv = k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


acc_NB = round(np.mean(score)*100,2)


# ## SVM

# In[ ]:


clf = SVC()
clf.fit(x_train, y_train)
y_pred_svc = clf.predict(x_test)
acc_svc = round(clf.score(x_train, y_train) * 100, 2)
print (acc_svc)


# ## Decision Tree

# In[ ]:


clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_predDTree = clf.predict(x_test)
accDTree = round(clf.score(x_train,y_train) *100, 2)
print(accDTree)


# ## Preceptron

# In[ ]:


clf = Perceptron(max_iter = 5, tol=None)
clf.fit(x_train,y_train)
y_predPercp = clf.predict(x_test)
accPercp = round(clf.score(x_train, y_train) * 100, 2)
print(accPercp)


# # Compare Model

# In[ ]:


models = pd.DataFrame({
    'Model':['KNN', 'Random Forest', 'Naive Bayes', 'SVM', 'Decision Tree','Perceptron'],
    'Score' : [accKNN, acc_RForest, acc_NB, acc_svc, accDTree, accPercp]
})
models.sort_values(by='Score', ascending=False)


# In[ ]:


test.head()


# In[ ]:


submission = pd.DataFrame({
    "PassengerId" : test["PassengerId"],
    "Survived" : y_predDTree
})


# In[ ]:


submission.to_csv('myResult.csv')


# # References
# 
# * Github
# * Titanic Beginnner Guide
