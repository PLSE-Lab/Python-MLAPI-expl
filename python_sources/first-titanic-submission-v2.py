#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from itertools import combinations

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_full = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
test_full = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')

train_NA = train_full.isna().sum()
test_NA = test_full.isna().sum()

pd.concat([train_NA, test_NA], axis=1, sort = False, keys = ['Train NA', 'Test NA'])


# # Eploratory Data Analysis
# 
# **Missing embark value**
# 
# The Titanic left Belfast on the 2nd of April, 1912. It made its way to Southampton, England then Cherbourg in France and finally Queenstown (Cork), Ireland. These are the 3 'Embarked' values: S, C, Q. 

# In[ ]:


train_full.Embarked.unique()


# In[ ]:


train_full[train_full.Embarked.isnull()]


# One passenger whose embark was unknown in the training data set was Mrs. George Nelson (Martha Evelyn) Stone and she is 62 years old. A quick Google search not only tells us that she embarked from Southampton, but that she was also a widowed first class passenger in cabin B28, and paid 80 for her fare which agrees with the data above. All we have to do is fill in her embark value with 'S'. The other passenger is Mrs. Stone's personal maid, Miss Amelie Icard from France. She also embarked from Southampton with Mrs. Stone. We also saw that they were both rescued on boat 6.

# In[ ]:


train_full.loc[830, 'Embarked'] = 'S'
train_full.loc[62, 'Embarked'] = 'S'
# print(train_full.loc[830], train_full.loc[62])


# In[ ]:


print('Passenger count from ')
print('Southampton: %d' %train_full[train_full.Embarked=='S'].Embarked.count())
print('Cherbourg: %d' %train_full[train_full.Embarked=='C'].Embarked.count())
print('Queenstown (Cork): %d' %train_full[train_full.Embarked=='Q'].Embarked.count())


# In[ ]:


print('Passenger count from (test)')
print('Southampton: %d' %test_full[test_full.Embarked=='S'].Embarked.count())
print('Cherbourg: %d' %test_full[test_full.Embarked=='C'].Embarked.count())
print('Queenstown (Cork): %d' %test_full[test_full.Embarked=='Q'].Embarked.count())


# In[ ]:


sns.catplot(x="Pclass", y="Fare", hue="Sex", kind="bar", data=train_full);
plt.title('Fare by class and gender (all)')
plt.ylabel('Fare')
plt.xlabel('Class')


# **Missing age values**

# In[ ]:


missing_age = train_full[train_full.Age.isnull()]
missing_age


# In[ ]:


sns.catplot(x="Pclass", y="Fare", hue="Sex", kind="bar", data=missing_age);
plt.title('Fare by class and gender (missing age)')
plt.ylabel('Fare')
plt.xlabel('Class')


# In[ ]:


plt.figure(figsize=(10,6))
plt.title('Missing age by class')

x = missing_age.Pclass
sns.countplot(x="Pclass", hue="Sex", data=missing_age)

plt.ylabel('Count')
plt.xlabel('Class')


# In[ ]:


plt.figure(figsize=(10,6))
plt.title('Missing age by embark')

x = missing_age.Pclass
sns.countplot(x="Embarked", hue="Sex", data=missing_age)

plt.ylabel('Count')
plt.xlabel('Embark')


# In[ ]:


med_age_S = train_full[(train_full.Embarked=='S')].Age.median()
med_age_C = train_full[(train_full.Embarked=='C')].Age.median()
med_age_Q = train_full[(train_full.Embarked=='Q')].Age.median()

print('Median age in ')
print('Southampton: %d' %med_age_S)
print('Cherbourg: %d' %med_age_C)
print('Queenstown (Cork): %d' %med_age_Q)


# In[ ]:


sns.relplot(x="Age", y="Fare", data=train_full);


# It doesn't seem like there is a relationship between age and fare, so we can't use fare to try and fill in age.

# In[ ]:


missing_age.info()


# In[ ]:


age_X = train_full[train_full.Age.notnull()]
# age_X.info()


# In[ ]:


age_X = age_X[age_X.columns.drop('Survived')]
age_X = age_X[age_X.columns.drop('Cabin')]


# In[ ]:


age_X.info()


# In[ ]:


age_y = age_X.Age.astype(int)
age_X = age_X[age_X.columns.drop('Age')]

ma_X, va_X, ma_y, va_y = train_test_split(age_X, age_y, random_state=1)

# ma_X.info()


# In[ ]:


# va_X.info()


# In[ ]:


# convert categorical features to numerical features using label encoder
le = LabelEncoder()

le_train_X = ma_X.copy()
le_valid_X = va_X.copy()

# Encode categorical features
s = ma_X.dtypes=='object'
cat_features = list(s[s].index)
# print(cat_features)

for col in cat_features:
    le_train_X[col] = le.fit_transform(ma_X[col])
    le_valid_X[col] = le.fit_transform(va_X[col])
    
ma_X = le_train_X
va_X = le_valid_X


# In[ ]:


# print(ma_X.info(), va_X.info())


# In[ ]:


rf_ma_model = RandomForestClassifier(n_estimators=100, random_state=0).fit(ma_X, ma_y)

pred_valid = rf_ma_model.predict(va_X)
print('Mean absolute error: %.2f' %mean_absolute_error(pred_valid, va_y))
print(accuracy_score(va_y, pred_valid))


# In[ ]:


missing_age = missing_age[missing_age.columns.drop('Survived')]
missing_age = missing_age[missing_age.columns.drop('Age')]
missing_age = missing_age[missing_age.columns.drop('Cabin')]


# In[ ]:


# convert categorical features to numerical features using label encoder
le = LabelEncoder()

le_train_X = missing_age.copy()

# Encode categorical features
s = missing_age.dtypes=='object'
cat_features = list(s[s].index)
print(cat_features)

for col in cat_features:
    le_train_X[col] = le.fit_transform(missing_age[col])
    
missing_age = le_train_X


# In[ ]:


missing_age.info()


# In[ ]:


pred_test = rf_ma_model.predict(missing_age)
output = pd.DataFrame({'Age': pred_test}, index=missing_age.index)
output


# In[ ]:


missing_age.info()


# In[ ]:


sns.distplot(pred_test, kde=False)


# In[ ]:


sns.distplot(train_full.Age, kde=False)


# In[ ]:


train_full[train_full.Age.isnull()].head()


# In[ ]:


test = train_full.copy()
test.head()


# In[ ]:


test = test.combine_first(output)
train_full = test
# train_full.info()


# # Cleaning the training and validation data set

# In[ ]:


y = train_full.Survived
X = train_full[train_full.columns.drop('Survived')]

train_X_full, valid_X_full, train_y, valid_y = train_test_split(X, y, random_state=1)


# **Dealing with missing 'Cabin' values**
# 
# One method would be to drop the 'Cabin' column. One discussion topic explained that first letter of the 'Cabin' related to the deck as well as the 'Pclass' of each passenger. Here, the first letters of the 'Cabin' column will be extracted and tested against the 'Pclass' of each passenger. The 'Cabin' will then be replaced by the letter that appears most frequent depending on the class of the passenger.

# In[ ]:


# Drop the cabin value
train_X = train_X_full[train_X_full.columns.drop('Cabin')]
valid_X = valid_X_full[valid_X_full.columns.drop('Cabin')]
# train_X.info()


# In[ ]:


train_X.head()


# In[ ]:


# convert categorical features to numerical features using label encoder
le = LabelEncoder()

le_train_X = train_X.copy()
le_valid_X = valid_X.copy()

# Encode categorical features
s = train_X.dtypes=='object'
cat_features = list(s[s].index)

for col in cat_features:
    le_train_X[col] = le.fit_transform(train_X[col])
    le_valid_X[col] = le.fit_transform(valid_X[col])
    
# le_train_X.info()


# In[ ]:


train_X = le_train_X
valid_X = le_valid_X


# In[ ]:


train_X.head()


# In[ ]:


# choose categorical features
cols = train_X.columns
interactions_train = pd.DataFrame(index=train_X.index)
interactions_valid = pd.DataFrame(index=valid_X.index)

# create the interactions for two categorical featuers at a time
for col1, col2 in combinations(cols, 2):
    col_name = '_'.join([col1, col2])
    
    values1 = train_X[col1].map(str) + '_' + train_X[col2].map(str)
    values2 = valid_X[col1].map(str) + '_' + valid_X[col2].map(str)
    
    encoder = preprocessing.LabelEncoder()
    
    interactions_train[col_name] = encoder.fit_transform(values1)
    interactions_valid[col_name] = encoder.fit_transform(values2)
    
interactions_train.head()


# In[ ]:


train_X = train_X.join(interactions_train)
# train_X.info()


# In[ ]:


valid_X = valid_X.join(interactions_valid)
# valid_X.info()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100,
                                  random_state=0).fit(train_X, train_y)


# In[ ]:


# for i in range(1,10):
#     rf_model = RandomForestClassifier(n_estimators=i*50,
#                                   random_state=0).fit(train_X, train_y)


# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rf_model, random_state=1).fit(valid_X, valid_y)
eli5.show_weights(perm, feature_names=valid_X.columns.tolist())


# In[ ]:


train_X.shape


# In[ ]:


from sklearn.metrics import mean_absolute_error

pred_valid = rf_model.predict(valid_X)
print('Mean absolute error: %.2f' %mean_absolute_error(pred_valid, valid_y))


# In[ ]:


from sklearn.metrics import accuracy_score

pred_valid = rf_model.predict(valid_X)
print(accuracy_score(valid_y, pred_valid))
print(pred_valid.tolist())


# In[ ]:


print(valid_y.tolist())


# In[ ]:


# # How many unique values of the top 10 features
# print('Gender and family (sibling/spouse): %d' %train_X.Sex_SibSp.nunique())
# print('Gender and embark location: %d' %train_X.Sex_Embarked.nunique())
# print('Class and embark location: %d' %train_X.Pclass_Embarked.nunique())
# print('Family (parent/children) and ticket: %d' %train_X.Parch_Ticket.nunique())
# print('Gender and family (parent/children): %d' %train_X.Sex_Parch.nunique())
# print('Class and gender: %d' %train_X.Pclass_Sex.nunique())
# print('Gender: %d' %train_X.Sex.nunique())
# print('Age and embark location: %d' %train_X.Age_Embarked.nunique())
# print('Age: %d' %train_X.Age.nunique())
# print('Ticket and fare cost: %d' %train_X.Ticket_Fare.nunique())

# print('\nNumber of passengers: %d \n  # of unique tickets: %d \n  people travelling alone: %d' %(len(train_X), 
#                                                                train_X.Ticket.nunique(), 
#                                                                len(train_X) - train_X.Ticket.nunique()))


# In[ ]:


print('Unique values of the top 10 features')
print('Gender and family (sibling/spouse): %d' %train_X.Sex_SibSp.nunique())
print('Gender: %d' %train_X.Sex.nunique())
print('Gender and ticket: %d' %train_X.Sex_Ticket.nunique())
print('Family (parent/children) and gender: %d' %train_X.Parch_Sex.nunique())
print('Age: %d' %train_X.Age.nunique())
print('Class and gender: %d' %train_X.Pclass_Sex.nunique())
print('Age and class: %d' %train_X.Age_Pclass.nunique())
print('Class and family (sibling/spouse): %d' %train_X.Pclass_SibSp.nunique())
print('Age and ticket: %d' %train_X.Age_Ticket.nunique())
print('Embark location and name: %d' %train_X.Embarked_Name.nunique())


# In[ ]:


# train_X.Fare.nunique()


# In[ ]:


plt.figure(figsize=(10,6))
plt.title('Age by gender/family (sibling/spouse)')

sns.barplot(x=train_X.Sex_SibSp, y=train_X.Age)

plt.xlabel('Family (sibling/spouse)')
plt.ylabel('Age')


# In[ ]:


# Male=1, Female=0
# Survived=1, not_survived=0
sns.swarmplot(y=train_X.SibSp, x=train_X.Sex, hue=train_y)


# In[ ]:


# Male=1, Female=0
# Survived=1, not_survived=0
sns.swarmplot(y=train_X.Parch, x=train_X.Sex, hue=train_y)


# # Cleaning up the test data set

# In[ ]:


test_full.info()


# In[ ]:


# Drop 'Cabin' from the test set
test_X = test_full[valid_X_full.columns.drop('Cabin')]

# Fill in missing embark values with the most frequent value of the test set
most_freq = test_X.Embarked.mode().iloc[0]
print(most_freq)
test_X.Embarked = test_X.Embarked.fillna(most_freq)


# In[ ]:


# test_X.info()


# In[ ]:


# convert categorical features to numerical features using label encoder
le = LabelEncoder()

le_test_X = test_X.copy()

# Encode categorical features
s = test_X.dtypes=='object'
cat_features = list(s[s].index)
print(cat_features)

for col in cat_features:
    le_test_X[col] = le.fit_transform(test_X[col])
    
# le_test_X.info()


# In[ ]:


test_X = le_test_X
# test_X.info()


# In[ ]:


missing_age = test_X[test_X.Age.isnull()]
missing_age = missing_age[missing_age.columns.drop('Age')]
missing_age.head()


# In[ ]:


pred_test = rf_ma_model.predict(missing_age)
output = pd.DataFrame({'Age': pred_test}, index=missing_age.index)
print(output.Age.tolist())
test_X = test_X.combine_first(output)
# test_X.info()


# In[ ]:


# choose categorical features
cols = test_X.columns
interactions_test = pd.DataFrame(index=test_X.index)

# create the interactions for two categorical featuers at a time
for col1, col2 in combinations(cols, 2):
    col_name = '_'.join([col1, col2])
    
    values = test_X[col1].map(str) + '_' + test_X[col2].map(str)
    
    encoder = preprocessing.LabelEncoder()
    
    interactions_test[col_name] = encoder.fit_transform(values)
    
interactions_test.head()


# In[ ]:


test_X = test_X.join(interactions_test)


# In[ ]:


test_X.head()


# In[ ]:


# Drop the missing fare value
test_X = test_X.fillna(0)


# In[ ]:


# test_X.info()


# In[ ]:


pred_test = rf_model.predict(test_X)
# pred_test


# In[ ]:


output = pd.DataFrame({'PassengerId': test_X.index,
                       'Survived': pred_test})
output.to_csv('titanic_pred2.csv', index=False)

