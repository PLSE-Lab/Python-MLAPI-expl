#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:



input_train = '../input/titanic/train.csv'
input_test = '../input/titanic/test.csv'

train = pd.read_csv(input_train)
test = pd.read_csv(input_test)

passengerId = test['PassengerId']


train.drop(labels='PassengerId', axis=1, inplace=True)
test.drop(labels='PassengerId', axis=1, inplace=True)


# In[ ]:


train.head(10)


# In[ ]:


train.columns


# In[ ]:


import pandas
from pandas.plotting import scatter_matrix

scatter_matrix(train, alpha=0.5, figsize=(20, 20))
plt.show()


# In[ ]:


train.hist(figsize=(20, 20), color='green')
plt.plot()


# In[ ]:


train.plot(subplots=True, figsize=(10, 10), sharex=False, sharey=False)
plt.show()


# In[ ]:


train.isnull().sum()


# In[ ]:


train.describe()


# In[ ]:


test.Fare.describe()


# In[ ]:


plt.subplots(figsize=(6,6))
sns.countplot('Survived', data=train,edgecolor=sns.color_palette('dark',5))
plt.xticks(rotation=90)
plt.title('Number Of Survived in Titanic')
plt.show()


# In[ ]:


plt.subplots(figsize=(10,6))
sns.countplot('Survived', hue='Sex',data=train,edgecolor=sns.color_palette('dark',5))
plt.xticks(rotation=90)
plt.title('Number Of Survived in Titanic')
plt.show()


# In[ ]:


onlyAge = train['Age']
plt.subplots(figsize=(16,6))
onlyAge.hist(bins=30,alpha=0.9)
plt.title('Histogram for Age')
plt.show()


# In[ ]:


sns.catplot(x="Pclass", hue="Sex", col="Survived", data=train, kind="count", height=6, aspect=.7)


# In[ ]:


train.Survived.value_counts().plot(kind='pie', autopct='%.2f%%')
plt.axis('equal')
plt.show()

train.Pclass.value_counts().plot(kind='pie', autopct='%.2f%%')
plt.axis('equal')
plt.show()

train.Sex.value_counts().plot(kind='pie', autopct='%.2f%%')
plt.axis('equal')
plt.show()

train.Embarked.value_counts().plot(kind='pie', autopct='%.2f%%')
plt.axis('equal')
plt.show()


# In[ ]:


plt.subplots(figsize=(18,6))
sns.countplot('Age',data=train,palette='RdYlGn',edgecolor=sns.color_palette('Paired',20),order=train['Age'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number of passengers separated by Age')
plt.show()


# In[ ]:


plt.subplots(figsize=(16,6))
sns.countplot('Survived', hue='Age',data=train,edgecolor=sns.color_palette('dark',5))
plt.xticks(rotation=90)
plt.title('Number Of Survived in Titanic by Age')
plt.show()


# In[ ]:


plt.subplots(figsize=(14,6))
sns.scatterplot(x="Age", y="Sex", hue="Survived", data=train)
plt.title("Relationship between age, sex and survivors")


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 6))
w = train[train['Sex']=='female']
m = train[train['Sex']=='male']

ax = sns.distplot(w[w['Survived']==1].Age.dropna(), bins=18, label ='label=1',ax = axes[0])
ax = sns.distplot(w[w['Survived']==0].Age.dropna(), bins=40, label ='label=0', ax = axes[0])
ax.legend()
ax.set_title('Female')

ax = sns.distplot(m[m['Survived']==1].Age.dropna(), bins=18, label = 'label=1', ax = axes[1])
ax = sns.distplot(m[m['Survived']==0].Age.dropna(), bins=40, label = 'label=0', ax = axes[1])
ax.legend()
ax = ax.set_title('Male')


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 6))
w = train[train['Sex']=='female']
m = train[train['Sex']=='male']

ax = sns.distplot(w[w['Survived']==1].Pclass.dropna(), bins=18, label ='label=1',ax = axes[0])
ax = sns.distplot(w[w['Survived']==0].Pclass.dropna(), bins=40, label ='label=0',ax = axes[0])
ax.legend()
ax.set_title('Female')

ax = sns.distplot(m[m['Survived']==1].Pclass.dropna(), bins=18, label = 'label=1',ax = axes[1])
ax = sns.distplot(m[m['Survived']==0].Pclass.dropna(), bins=40, label = 'label=0',ax = axes[1])
ax.legend()
ax = ax.set_title('Male')


# In[ ]:


grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=4, aspect=1.6)
grid.map(plt.hist,'Age', alpha=0.9, bins=20, color='red')
grid.add_legend();


# In[ ]:


plt.subplots(figsize=(10,6))
sns.countplot('Survived', hue='Pclass',data=train, edgecolor=sns.color_palette('dark',5))
plt.xticks(rotation=90)
plt.title('Number Of Survived in Titanic by PClass')
plt.show()


# In[ ]:


train['Sex'] = train['Sex'].map({"male":0,"female":1})
test['Sex'] = test['Sex'].map({"male":0,"female":1})
train['Embarked'] = train['Embarked'].map({"S":0,"C":1,"Q":2})
test['Embarked'] = test['Embarked'].map({"S":0,"C":1,"Q":2})

def deleteTag(train, test):
    delete = ['Name', 'Ticket', 'Cabin']
    for name in delete:
        train = train.drop([name], axis=1)
        test = test.drop([name], axis=1)
        
    return [train, test]

train, test = deleteTag(train, test)

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

X = train.iloc[:, 1:12].values
y = train.iloc[:, 0].values

s = LabelEncoder()
t = LabelEncoder()
e = LabelEncoder()
X[:, 1] = s.fit_transform(X[:, 1])
X[:, 5] = t.fit_transform(X[:, 5])
X[:, 4] = e.fit_transform(X[:, 4])

print(X)

X = imp.fit_transform(X)


# In[ ]:


test = imp.fit_transform(test)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.17)


# In[ ]:


X_train


# In[ ]:


RandomForest = RandomForestClassifier(n_estimators=20, criterion='entropy')
RandomForest.fit(X_train, y_train)


# In[ ]:


print(test)


# In[ ]:


prediction = RandomForest.predict(test)
print(prediction)


# In[ ]:


print(len(prediction))


# In[ ]:


Kagglesubmission = pd.DataFrame({'PassengerId': passengerId, 'Survived': prediction})


# In[ ]:


Kagglesubmission.to_csv('rdmf_Titanic.csv', index=False)

