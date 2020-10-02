#!/usr/bin/env python
# coding: utf-8

# This notebook is written in Python
# Reference - "Titanic best working Classifier" - https://www.kaggle.com/sinakhorami/titanic-best-working-classifier.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import re as re
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('/kaggle/input/titanic/test.csv' , header = 0, dtype={'Age': np.float64})
full_data = [train, test]


# ### Feature Engineering

# In[ ]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean()


# In[ ]:


train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean()


# In[ ]:


# create new feature called Family Size
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean()


# In[ ]:


# new feature called IsAlone to check if being single has any effect on the survival
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=True).mean()


# In[ ]:


# fill the NaN with some values for 'Embarked'Randomly
for dataset in full_data:
    choice = random.choice(['C','S','Q'])
    dataset['Embarked'] = dataset['Embarked'].fillna(choice)
    
print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())


# In[ ]:


# Pclass vs Fare is it connsistent? NO
# so lets not use Fare as a feature.
grouped = train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=True)
# for name, group in grouped:
#     print(name)
#     print(group)


# In[ ]:


# fill the Fare NaN's using Pclass
dataset_name = ['train', 'test']
# print(test[test['Fare'].isnull()].index)  # only one index in test and 0 in train

for name, dataset in zip(dataset_name, full_data):
    grouped = dataset[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False)
    ndex = dataset[dataset['Fare'].isnull()].index.tolist()
    print('Dataset - "{}" Index with Null Value - {}'.format(name, ndex))
    for i in ndex:
        print("Current Value - ", dataset.loc[i, 'Fare'])
        pclass = dataset.loc[i, 'Pclass']
        dataset.loc[i, 'Fare'] = grouped.get_group(pclass).mean()['Fare']
        print("Updated Value - ", dataset.loc[i, 'Fare'])


# In[ ]:


# Fill in Age with values within Gaussian distribution and Categorize the age data
# first fill the NaN's and then convert the age to int
for dataset in full_data:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    fill_age_value = np.random.randint(age_avg-age_std, age_avg+age_std, size=age_null_count)
    dataset.loc[dataset['Age'].isnull(), 'Age'] = fill_age_value
    dataset['Age'] = dataset['Age'].astype(int)


# In[ ]:


# convert to catergorical data and do impact analysis on Survivals
train['CategoricalAge'] = pd.cut(train['Age'], 5)
train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=True).mean()


# In[ ]:


# With Name we can find the title of people
def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name) 
    if title_search:
        return title_search.group(1) # group1 is just the (value within this) without the .
    return ""
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)


# In[ ]:


pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[ ]:


train[['Title', 'Survived']].groupby(['Title'], as_index=True).mean()


# ## Data Cleaning

# In[ ]:


train.head(2)


# In[ ]:


for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map({'female':0, 'male':1}).astype(int)
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# ### Feature Selection

# In[ ]:


drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize', 'Fare']
train = train.drop(drop_elements, axis = 'columns')
test_passengerId = test['PassengerId']
test = test.drop(drop_elements, axis='columns')
train = train.drop(['CategoricalAge'], axis = 'columns')


# ### Classifier Comparision

# In[ ]:


classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
	AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression() ]
log_cols = ["Classifier", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)


# In[ ]:


X = train.drop(['Survived'], axis='columns')
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)    
    acc = accuracy_score(y_test, train_predictions)
    log = log.append({'Classifier':name, 'Accuracy':acc}, ignore_index=True)

plt.title('Classifier Accuracy')
plt.xlabel('Accuracy')
# sns.set_color_codes("muted")
sns.barplot(x="Accuracy", y="Classifier", data=log, color='g')


# ### Real Test Prediction

# In[ ]:


clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(test)


# In[ ]:


output = pd.DataFrame({'PassengerID': test_passengerId, 'Survived': predictions })
output.to_csv('my_submissions.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




