#!/usr/bin/env python
# coding: utf-8

# # index
# ---
# 1. [load libraries and datasets](#load)
# 2. [glimpse dataset](#glimpse)
# 3. [outlier detection](#outlier)
# 4. [preprocess data](#preprocess)
# 5. [train](#train)
# 6. [predict](#predict)

# # <a id='load'>1. load libraries and datasets</a>

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tpot import TPOTClassifier


# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv') 
test_df = pd.read_csv('../input/titanic/test.csv')

print('train_df.shape:', train_df.shape)
print('test_df.shape:', test_df.shape)


# In[ ]:


# combine train, test data
train_test_data = [train_df, test_df]


# # <a id='glimpse'>2. glimpse dataset</a>

# In[ ]:


print(train_df.columns.to_list())
print(test_df.columns.to_list())


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe(include='object')


# In[ ]:


test_df.info()


# In[ ]:


test_df.describe(include='object')


# In[ ]:


# null field
train_null_s = train_df.isnull().sum()
print(train_null_s[train_null_s != 0])
print('-'*80)
test_null_s = test_df.isnull().sum()
print(test_null_s[test_null_s != 0])


# # <a id="outlier">3. outlier detection</a>
#   - **target field**: `Age`, `Fare`
#   - exclude detection field
#     - object type: `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`
#     - simple index field: `PassengerId`
#     - well-refined field: `Pclass`, `SibSp`, `Parch`
#   - drop outlier record

# In[ ]:


# detect target outlier index
outlier_detection_field = ['Age', 'Fare']
weight = 2

outlier_indices = []

for col in outlier_detection_field:
    q1 = np.nanpercentile(train_df[col], 25)
    q3 = np.nanpercentile(train_df[col], 75)
    iqr = q3-q1
    iqr_weight = iqr * weight

    lowest_val = q1 - iqr_weight
    highest_val = q3 + iqr_weight

    outlier_index = train_df[(train_df[col]<lowest_val) | (highest_val<train_df[col])].index
    outlier_indices.extend(outlier_index)
    
    print('{}: {} / {} (record size:{})'.format(col, lowest_val, highest_val, outlier_index.shape[0]))


# In[ ]:


# drop outlier index
train_df.drop(outlier_indices, axis=0, inplace=True)


# # <a id="preprocess">4. preprocess data</a>
#   - [`PassengerId`](#PassengerId)
#   - [`Pclass`](#Pclass)
#   - [`Name`](#Name)
#   - [`Sex`](#Sex)
#   - [`Age`(missing value exists)](#Age)
#   - [`SibSp`, `Parch`](#SibSp)
#   - [`Ticket`](#Ticket)
#   - [`Fare`(missing value exists)](#Fare)
#   - [`Cabin`(missing value exists)](#Cabin)
#   - [`Embarked`(missing value exists)](#Embarked)

# ### <a id="PassengerId">`PassengerId` field</a>
#   - drop

# In[ ]:


train_df['PassengerId']


# In[ ]:


test_df['PassengerId']


# In[ ]:


# drop 'passengerId' field
# for test data, save the 'PassengerId' field for submission

train_df.drop('PassengerId', axis=1, inplace=True)

test_df_PId = test_df['PassengerId']
test_df.drop('PassengerId', axis=1, inplace=True)


# In[ ]:


print(train_df.columns.to_list())
print(test_df.columns.to_list())


# ### <a id='Pclass'>`Pclass` field</a>
#   - MinMaxScale

# In[ ]:


print(train_df['Pclass'].value_counts())

sns.barplot(data=train_df, x='Pclass', y='Survived')


# In[ ]:


# scale
minMaxScaler = MinMaxScaler()

for data in train_test_data:
    data['Pclass'] = minMaxScaler.fit_transform(data[['Pclass']])


# In[ ]:


train_df['Pclass'].value_counts()


# ### <a id='Name'>`Name` field</a>
#   - get 'Title' field from 'Name' field
#   - encode
#   - MinMaxScale

# In[ ]:


train_df['Name'].head(10)


# In[ ]:


# get 'title' field from 'name' field

# train_df['Name'].str.extract(' ([a-zA-Z]+)\. ', expand=False).value_counts()
for data in train_test_data:
    data['Title'] = data['Name'].str.extract(' ([a-zA-Z]+)\. ', expand=False)


# In[ ]:


# drop 'name' field

for data in train_test_data:
    data.drop('Name', axis=1, inplace=True)


# In[ ]:


print(train_df['Title'].value_counts())
print('-'*50)
print(test_df['Title'].value_counts())


# In[ ]:


# encode

title_mapping = {
    'Mr':0,
    'Miss':1,
    'Mrs':2,
    'Master':3,
    'Dr':4, 'Rev':4, 'Major':4, 'Mlle':4, 'Col':4, 'Ms':4, 'Countess':4, 'Mme':4, 'Lady':4, 'Sir':4, 'Don':4, 'Jonkheer':4, 'Capt':4, 'Dona':4
}

for data in train_test_data:
    data['Title'] = data['Title'].map(title_mapping)


# In[ ]:


train_df['Title'].value_counts()


# In[ ]:


# scale

minMaxScaler = MinMaxScaler()

for data in train_test_data:
    data['Title'] = minMaxScaler.fit_transform(data[['Title']])


# In[ ]:


train_df['Title'].value_counts()


# In[ ]:


sns.barplot(data=train_df, x='Title', y='Survived')


# ### <a id='Sex'>`Sex` field</a>
#   - encode

# In[ ]:


print(train_df['Sex'].value_counts())

sns.barplot(data=train_df, x='Sex', y='Survived')


# In[ ]:


# encode

for data in train_test_data:
    data['Sex'] = data['Sex'].astype('category').cat.codes


# In[ ]:


train_df['Sex'].value_counts()


# ### <a id='Age'>`Age` field</a>
#   - missing value exists
#   - binning
#   - MinMaxScale

# In[ ]:


train_df['Age'].isnull().sum()


# In[ ]:


# fill null with the middle value of the title
# train_df.groupby('Title')['Age'].transform('median')

for data in train_test_data:
    data['Age'].fillna(train_df.groupby('Title')['Age'].transform('median'), inplace=True)


# In[ ]:


train_df['Age'].isnull().sum()


# In[ ]:


sns.distplot(train_df['Age'])


# In[ ]:


# binning
# pd.qcut(train_df['Age'], 5).cat.codes

for data in train_test_data:
    data['Age'] = pd.qcut(data['Age'], 9).cat.codes


# In[ ]:


# scale

minMaxScaler = MinMaxScaler()

for data in train_test_data:
    data['Age'] = minMaxScaler.fit_transform(data[['Age']])


# In[ ]:


train_df['Age'].describe()


# ### <a id='SibSp'>`SibSp`, `Parch` field</a>
#   - get 'FamilySize' from 'SibSp', 'Parch' field
#   - drop 'SibSp', 'Parch' field
#   - ~~binning~~
#   - MinMaxScale

# In[ ]:


print(train_df['SibSp'].value_counts())
sns.barplot(data=train_df, x='SibSp', y='Survived')


# In[ ]:


print(train_df['Parch'].value_counts())
sns.barplot(data=train_df, x='Parch', y='Survived')


# In[ ]:


for data in train_test_data:
    data['FamilySize'] = data['Parch'] + data['SibSp']


# In[ ]:


sns.barplot(data=train_df, x='FamilySize', y='Survived')


# In[ ]:


# drop
for data in train_test_data:
    data.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# In[ ]:


# binning
# train_df.loc[(1<=train_df['FamilySize']) & (train_df['FamilySize']<4), 'FamilySize'].value_counts()

# for data in train_test_data:
#     data.loc[data['FamilySize']==0, 'FamilySize'] = 0
#     data.loc[(1<=data['FamilySize']) & (data['FamilySize']<4), 'FamilySize'] = 1
#     data.loc[(4<=data['FamilySize']) & (data['FamilySize']<7), 'FamilySize'] = 2
#     data.loc[(7<=data['FamilySize']), 'FamilySize'] = 3


# In[ ]:


# scale
minMaxScaler = MinMaxScaler()

for data in train_test_data:
    data['FamilySize'] = minMaxScaler.fit_transform(data[['FamilySize']])


# In[ ]:


sns.barplot(data=train_df, x='FamilySize', y='Survived')


# ### <a id='Ticket'>`Ticket` field</a>
#   - drop

# In[ ]:


print(train_df['Ticket'].value_counts())
print('-'*80)
print(train_df['Ticket'].unique().shape)


# In[ ]:


# drop 'Ticket' field
for data in train_test_data:
    data.drop('Ticket', axis=1, inplace=True)


# ### <a id='Fare'>`Fare` field</a>
#   - missing value exists
#   - log transformation to improve skewed data
#   - binning
#   - MinMaxScale

# In[ ]:


print(train_df['Fare'].isnull().sum())
print(test_df['Fare'].isnull().sum())


# In[ ]:


train_df.groupby(['Embarked', 'Pclass'])['Fare'].median()


# In[ ]:


# fill null with the middle value of the 'Embarked', 'Pclass'

for data in train_test_data:
    data['Fare'].fillna(train_df.groupby(['Embarked', 'Pclass'])['Fare'].transform('median'), inplace=True)


# In[ ]:


test_df['Fare'].isnull().sum()


# In[ ]:


sns.distplot(train_df['Fare'])


# In[ ]:


# log transformation to import skewed data
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

sns.distplot(train_df['Fare'], ax=ax1)
sns.distplot(np.log1p(train_df['Fare']), ax=ax2)

for data in train_test_data:
    data['Fare'] = np.log1p(data[['Fare']])


# In[ ]:


# binning
# pd.qcut(train_df['Fare'], 5).astype('category').cat.codes.value_counts()

for data in train_test_data:
    data['Fare'] = pd.qcut(data['Fare'], 10).astype('category').cat.codes


# In[ ]:


train_df['Fare'].value_counts()


# In[ ]:


# scale
minMaxScaler = MinMaxScaler()

for data in train_test_data:
    data['Fare'] = minMaxScaler.fit_transform(data[['Fare']])


# In[ ]:


train_df['Fare'].value_counts()


# ### <a id='Cabin'>`Cabin` field</a>
#   - missing value exists
#   - replace field to the first character of the field
#   - encode
#   - MinMaxScale

# In[ ]:


print(train_df['Cabin'].value_counts())
print('-'*80)
print(train_df['Cabin'].unique().shape)
print('-'*80)
print(train_df['Cabin'].str[:1].value_counts())


# In[ ]:


print(test_df['Cabin'].str[:1].value_counts())


# In[ ]:


# replace 'Cabin' field to the first character of the field
for data in train_test_data:
    data['Cabin'] = data['Cabin'].str[:1]


# In[ ]:


sns.barplot(data=train_df, x='Cabin', y='Survived')


# In[ ]:


# encode
cabin_mapping={"A":1, "B":2, "C":3, "D":4, "E":5, "F":6, "G":7, "T":8}

for data in train_test_data:
    data['Cabin'] = data['Cabin'].map(cabin_mapping)


# In[ ]:


print(train_df['Cabin'].value_counts())
print('-'*80)
# print(train_df.groupby(['Pclass', 'Embarked'])['Cabin'].median())
print(train_df.groupby(['Pclass'])['Cabin'].median())


# In[ ]:


# fill null with the middle value of the 'Pclass'
for data in train_test_data:
    data['Cabin'].fillna(data.groupby(['Pclass'])['Cabin'].transform('median'), inplace=True)


# In[ ]:


print(train_df['Cabin'].isnull().sum())
print('-'*80)
print(train_df['Cabin'].value_counts())


# In[ ]:


# scale
minMaxScaler = MinMaxScaler()

for data in train_test_data:
    data['Cabin'] = minMaxScaler.fit_transform(data[['Cabin']])


# ### <a id='Embarked'>`Embarked` field</a>
#   - missing value exists
#   - encode
#   - MinMaxScale

# In[ ]:


print(train_df['Embarked'].isnull().sum())
print(test_df['Embarked'].isnull().sum())


# In[ ]:


print(train_df['Embarked'].value_counts())
sns.barplot(data=train_df, x='Embarked', y='Survived')


# In[ ]:


# there aren't many missing values(just 2 records in train data), so fill null to most value
for data in train_test_data:
    data['Embarked'] = data['Embarked'].fillna('S')


# In[ ]:


# encode

for data in train_test_data:
    data['Embarked'] = data['Embarked'].astype('category').cat.codes


# In[ ]:


# scale
minMaxScaler = MinMaxScaler()

for data in train_test_data:
    data['Embarked'] = minMaxScaler.fit_transform(data[['Embarked']])


# # <a id="train">5. train</a>
#   - [classifier cross_val_score](#cross_val_score)
#   - [hyperparameter tuning](#tuning)

# In[ ]:


train_df.head()


# In[ ]:


y_train_s = train_df['Survived']
x_train_df = train_df.drop('Survived', axis=1)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_train_df, y_train_s, test_size=0.2, random_state=10)


# ### <a id='cross_val_score'>classifier cross_val_score</a>

# In[ ]:


def cross_val_score_result(estimator, x, y, scoring, cv):
    clf_scores = cross_val_score(estimator, x, y, scoring=scoring, cv=cv)
    clf_scores_mean = np.round(np.mean(clf_scores), 4)
    
    return clf_scores_mean


# In[ ]:


classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    KNeighborsClassifier(),
    SVC(),
    LGBMClassifier(),
    XGBClassifier(), 
    AdaBoostClassifier()
#     TPOTClassifier()
]


# In[ ]:


best_clf_score = 0
best_clf = None

clf_name = []
clf_mean_score = []

for clf in classifiers:
    current_clf_score = cross_val_score_result(clf, x_train, y_train, 'accuracy', 10)
    clf_name.append(clf.__class__.__name__)
    clf_mean_score.append(current_clf_score)
    
    if current_clf_score > best_clf_score:
        best_clf_score = current_clf_score
        best_clf = clf


# In[ ]:


clf_df = pd.DataFrame({"clf_name":clf_name, "clf_mean_score":clf_mean_score})
plt.figure(figsize=(8, 6))
sns.barplot(data=clf_df, x="clf_mean_score", y="clf_name")

print('best classifier: {}({})'.format(best_clf.__class__.__name__, best_clf_score))


# ### <a id="tuning">classifier hyperparameter tuning</a>

# In[ ]:


# train the classifier get the highest score
lgbm_clf = LGBMClassifier()

grid_param = {
    'learning_rate':[0.005, 0.01, 0.015, 0.02],
    'n_estimators':[100, 150, 200],
    'bossting_type':['rf', 'gbdt', 'dart', 'goss'],
    'max_depth':[10, 15, 20]
}

lgbm_grid = GridSearchCV(lgbm_clf, grid_param, cv=10)
lgbm_grid.fit(x_train, y_train)


# In[ ]:


print('best_param:', lgbm_grid.best_params_)
print('best_score:{:.4f}'.format(lgbm_grid.best_score_))


# # <a id="predict">6. predict</a>

# In[ ]:


test_df.head()


# In[ ]:


test_pred = lgbm_grid.best_estimator_.predict(test_df)

submission = pd.DataFrame({
    'PassengerId': test_df_PId,
    'Survived': test_pred
})


# In[ ]:


submission.to_csv('submission_test.csv', index=False)


# In[ ]:


check_submission = pd.read_csv('submission_test.csv')
check_submission

