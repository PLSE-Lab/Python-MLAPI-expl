#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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


pd.set_option('display.max_rows', None)


# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()


# In[ ]:


train_data.info()


# **1. Preprocessing train data**

# * Title

# In[ ]:


train_data['Name'] = train_data['Name'].apply(lambda x : x.split(',')[1].split('.')[0].strip())
train_data.groupby('Name')['PassengerId'].nunique()


# In[ ]:


titles = {
    'Capt' : 'Officer',
    'Col'  : 'Officer',
    'Dr'   : 'Officer',
    'Major': 'Officer',
    'Rev'  : 'Officer',
    'Jonkheer': 'Royalty',
    'Don'     : 'Royalty',
    'Dona'    : 'Royalty',
    'Lady'    : 'Royalty',
    'Sir'     : 'Royalty',
    'the Countess': 'Royalty',
    'Mr'  : 'Mr',
    'Mrs' : 'Mrs',
    'Ms'  : 'Mrs',
    'Mme' : 'Mrs',
    'Miss': 'Miss',
    'Mlle': 'Miss',
    'Master': 'Master'
}

train_data['Name'] = train_data['Name'].map(titles)


# * Ticket

# In[ ]:


room_occupant = train_data.groupby('Ticket')['PassengerId'].nunique()
train_data = train_data.assign(Occupancy=train_data['Ticket'].map(room_occupant))
train_data.drop(['Ticket'], axis=1, inplace=True)


# * Cabin<br>

# In[ ]:


train_data['Cabin'].fillna('M', inplace=True)
train_data['Cabin'] = train_data['Cabin'].apply(lambda x: x[0])

train_data['Cabin'].value_counts()


# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt

survivor_count = train_data.groupby('Cabin')['Survived'].apply(lambda x: x[x==1].count())
tuples = list(zip(['A','B','C','D','E','F','G','M','T'], survivor_count))
bar_df = pd.DataFrame(tuples, columns = ['Cabin', 'SurvivorCount'])

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f.set_figheight(5)
f.set_figwidth(10)
sns.barplot(x='Cabin', y='SurvivorCount', data=bar_df, palette='husl', ax=ax1)

death_count = train_data.groupby('Cabin')['Survived'].apply(lambda x: x[x==0].count())
tuples = list(zip(['A','B','C','D','E','F','G','M','T'], death_count))
bar_df = pd.DataFrame(tuples, columns = ['Cabin', 'DeathCount'])

sns.barplot(x='Cabin', y='DeathCount', data=bar_df, palette='husl', ax=ax2)


# In[ ]:


train_data.groupby(['Cabin','Pclass'])['PassengerId'].count()


# In[ ]:


train_data['Cabin'] = train_data['Cabin'].map({'A':'ABC', 'B':'ABC', 'C':'ABC', 'D':'DE', 'E':'DE', 'F':'FG', 'G':'FG', 'M':'M', 'T':'ABC'})

dummy = pd.get_dummies(train_data['Cabin'])
train_data.drop('Cabin', axis=1, inplace=True)
train_data = pd.concat([train_data, dummy], axis = 1)


# * Embarked

# In[ ]:


train_data.loc[train_data['Embarked'].isnull()]


# We can fill the missing values with the most frequently occuring port.

# In[ ]:


mode = train_data['Embarked'].mode()[0]
train_data['Embarked'].fillna(mode, inplace=True)

dummy = pd.get_dummies(train_data['Embarked'])
train_data.drop('Embarked', axis=1, inplace=True)
train_data = pd.concat([train_data, dummy], axis = 1)


# * Age<br>

# In[ ]:


train_data.groupby(['Pclass','Name','Sex'])['Age'].median()


# In[ ]:


median_df = train_data.groupby(['Pclass','Name','Sex'])['Age'].median().to_frame()
train_data['Age'] = train_data.groupby(['Pclass', 'Name', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))


# * Sex

# In[ ]:


dummy = pd.get_dummies(train_data['Sex'])
train_data = pd.concat([train_data, dummy], axis = 1)
train_data.drop('Sex', axis=1, inplace=True)


# * Name

# In[ ]:


dummy = pd.get_dummies(train_data['Name'])
train_data = pd.concat([train_data, dummy], axis = 1)
train_data.drop('Name', axis=1, inplace=True)


# In[ ]:


data = train_data[train_data.columns.difference(['PassengerId'])]
plt.figure(figsize=(30,15))
ax = sns.heatmap(data.corr(), annot=True)


# In[ ]:


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


# In[ ]:


test_data.info()


# **2. Preprocessing test data**

# * Fare

# In[ ]:


test_data.loc[test_data['Fare'].isnull()]


# In[ ]:


third_class = test_data.loc[test_data['Pclass'] == 3]
mean_fare = third_class['Fare'].mean()
test_data['Fare'].fillna(mean_fare, inplace=True)


# * Title

# In[ ]:


test_data['Name'] = test_data['Name'].apply(lambda x : x.split(',')[1].split('.')[0].strip())
test_data.groupby('Name')['PassengerId'].nunique()


# In[ ]:


test_data['Name'] = test_data['Name'].map(titles)

dummy = pd.get_dummies(test_data['Name'])
test_data = pd.concat([test_data, dummy], axis = 1)


# * Cabin

# In[ ]:


test_data['Cabin'].fillna('M', inplace=True)
test_data['Cabin'] = test_data['Cabin'].apply(lambda x: x[0])

test_data['Cabin'] = test_data['Cabin'].map({'A':'ABC', 'B':'ABC', 'C':'ABC', 'D':'DE', 'E':'DE', 'F':'FG', 'G':'FG', 'M':'M', 'T':'ABC'})

dummy = pd.get_dummies(test_data['Cabin'])
test_data.drop('Cabin', axis=1, inplace=True)
test_data = pd.concat([test_data, dummy], axis = 1)


# * Age

# In[ ]:


age_is_null = test_data.loc[test_data['Age'].isnull(), ['Pclass','Name','Sex']]
filled_age = age_is_null.reset_index().merge(median_df, on=['Pclass','Name','Sex'], how='left').set_index('index')
predicted_age = dict(zip(filled_age.index, filled_age['Age']))
test_data['Age'].fillna(predicted_age, inplace=True)


# * Sex

# In[ ]:


dummy = pd.get_dummies(test_data['Sex'])
test_data.drop('Sex', axis=1, inplace=True)
test_data = pd.concat([test_data, dummy], axis = 1)


# * Name and Ticket

# In[ ]:


room_occupant = test_data.groupby('Ticket')['PassengerId'].nunique()
test_data = test_data.assign(Occupancy=test_data['Ticket'].map(room_occupant))
test_data.drop(['Ticket', 'Name'], axis=1, inplace=True)


# * Embarked

# In[ ]:


dummy = pd.get_dummies(test_data['Embarked'])
test_data.drop('Embarked', axis=1, inplace=True)
test_data = pd.concat([test_data, dummy], axis = 1)


# **3. Train model**

# In[ ]:


target = train_data['Survived'].to_numpy()
train_predictor = train_data[train_data.columns.difference(['PassengerId', 'Survived'])].to_numpy()
test_predictor = test_data[test_data.columns.difference(['PassengerId'])].to_numpy()


# In[ ]:


from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
scaled_train = scaler.fit_transform(train_predictor)
scaled_test = scaler.fit_transform(test_predictor)


# In[ ]:


from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix

skf = StratifiedKFold(n_splits=10, random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

estimator = RandomForestClassifier(random_state=42, max_depth=10, min_samples_leaf=2, min_samples_split=4)
n_estimators = [n for n in range(100, 1000, 100)]
param_grid = {'n_estimators': n_estimators}
rfc = GridSearchCV(estimator, param_grid, cv=3)

scores = cross_val_score(rfc, train_predictor, target, cv=skf)
print(scores)
print('Average score :', np.mean(scores))
y_pred = cross_val_predict(rfc, train_predictor, target, cv=10)
sns.heatmap(confusion_matrix(target, y_pred),annot=True,fmt='3.0f')
plt.title('Confusion_matrix', y=1.05, size=15)


# In[ ]:


from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(max_iter=10000, random_state=42)
penalty = ['l1', 'l2']
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = {'C': C, 'penalty': penalty}
lr = GridSearchCV(estimator, param_grid, cv=3)

scores = cross_val_score(lr, scaled_train, target, cv=skf)
print(scores)
print('Average score :', np.mean(scores))
y_pred = cross_val_predict(lr, scaled_train, target, cv=10)
sns.heatmap(confusion_matrix(target, y_pred),annot=True,fmt='3.0f')
plt.title('Confusion_matrix', y=1.05, size=15)


# In[ ]:


from sklearn.svm import SVC

estimator = SVC(random_state=42)
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
svm = GridSearchCV(estimator, param_grid, cv=3)

scores = cross_val_score(svm, scaled_train, target, cv=skf)
print(scores)
print('Average score :', np.mean(scores))
y_pred = cross_val_predict(svm, scaled_train, target, cv=10)
sns.heatmap(confusion_matrix(target, y_pred),annot=True,fmt='3.0f')
plt.title('Confusion_matrix', y=1.05, size=15)


# **4. Make predictions using the best model**

# In[ ]:


svm.fit(scaled_train, target)
predictions = svm.predict(scaled_test)

