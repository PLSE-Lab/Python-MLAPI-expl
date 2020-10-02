#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
train_data


# In[ ]:


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


#checking if the guess of females survived in gender_subbmission.csv is reasonable

women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women)/len(women)
print("Percentage of women who survived: ", rate_women)


# In[ ]:


men = train_data.loc[train_data.Sex == 'male']['Survived']
rate_men = sum(men)/len(men)
print("Percentage of men who survived: ", rate_men)


# In[ ]:


#74% female survived and 18% men survived; means that the the assumption in gender_submission.csv is not a bad guess


# In[ ]:


#sns.countplot(train_data['Embarked'])
sns.countplot(x = 'Embarked', data = train_data)


# # Passenger ID

# In[ ]:


train_data.PassengerId.nunique()
passengerId = test_data['PassengerId']


# In[ ]:


train_data.head()


# In[ ]:


train_data.drop(labels='PassengerId', axis=1, inplace=True)
test_data.drop(labels='PassengerId', axis=1, inplace=True)


# # PClass

# In[ ]:


fx, axes = plt.subplots(1, 2, figsize=(15,6))
axes[0].set_title("Pclass vs Frequency")
axes[1].set_title("Pclass vs Survival rate")
fig1_pclass = sns.countplot(data=train_data, x='Pclass', ax=axes[0])
fig2_pclass = sns.barplot(data=train_data, x='Pclass',y='Survived', ax=axes[1])


# In[ ]:


train_data['Survived'].groupby(train_data['Pclass']).mean()


# In[ ]:


train_data['Name'].nunique()


# In[ ]:


train_data.head()


# # Name

# In[ ]:


train_data['Title'] = train_data['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
test_data['Title'] = train_data['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

train_data['Name_Len'] = train_data['Name'].apply(lambda x: len(x))
test_data['Name_Len'] = test_data['Name'].apply(lambda x: len(x))

train_data.drop('Name', axis=1, inplace=True)
test_data.drop('Name', axis=1, inplace=True)


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


test_data.Name_Len = (test_data.Name_Len/10).astype(np.int64)+1
train_data.Name_Len = (train_data.Name_Len/10).astype(np.int64)+1


# In[ ]:


train_data['Survived'].groupby(train_data['Title']).mean()


# In[ ]:


train_data['Title']


# In[ ]:


train_data['Survived'].groupby(train_data['Name_Len']).mean()


# In[ ]:


fx, axes = plt.subplots(2, 1, figsize=(15, 10))
axes[0].set_title("Title vs Frequency")
axes[1].set_title("Title vise Survival rate")
fig1_title = sns.countplot(data=train_data, x='Title', ax=axes[0])
fig2_title = sns.barplot(data=train_data, x='Title',y='Survived', ax=axes[1])


# # Gender

# In[ ]:


fx, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].set_title("Gender vs Frequency")
axes[1].set_title("Gender vise Survival rate")
fig1_gen = sns.countplot(data=train_data, x='Sex', ax=axes[0])
fig2_gen = sns.barplot(data=train_data, x='Sex', y='Survived', ax=axes[1])


# In[ ]:


train_data['Survived'].groupby(train_data['Sex']).mean()


# In[ ]:


genders = {'male': 0, 'female': 1}
data = [train_data, test_data]
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)


# # Age

# In[ ]:


print(train_data['Age'].isnull().sum())
print(test_data['Age'].isnull().sum())
train_data.info()


# In[ ]:


#training_age_n = train_data['Age'].dropna(axis=0)
my_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
train_data['Age'] = my_imputer.fit_transform(train_data[['Age']]).ravel()
test_data['Age'] = my_imputer.transform(test_data[['Age']]).ravel()
training_age_n = train_data


# fx, axes = plt.subplots(1, 2, figsize=(15,5))
# axes[0].set_title("Age vs frequency")
# axes[1].set_title("Age vise Survival rate")
# fig1_age = sns.distplot(a=training_age_n, bins=15, ax=axes[0], hist_kws={'rwidth':0.7})
# 
# # Creating a new list of survived and dead
# 
# pass_survived_age = train_data[train_data['Survived'] == 1].Age
# pass_dead_age = train_data[train_data['Survived'] == 0].Age
# 
# axes[1].hist([train_data.Age, pass_survived_age, pass_dead_age], bins=5, range=(0, 100), label=['Total', 'Survived', 'Dead'])
# axes[1].legend()
# plt.show()

# full_data = pd.concat([train_data, test_data])
# full_data.head()

# train_age_mean = full_data.Age.mean()
# train_age_std = full_data.Age.std()
# train_age_null = train_data.Age.isnull().sum()
# rand_tr_age = np.random.randint(train_age_mean - train_age_std, train_age_mean + train_age_std, size=train_age_null)
# train_data['Age'][np.isnan(train_data['Age'])] = rand_tr_age
# train_data['Age'] = train_data['Age'].astype(int) + 1
# 
# # Null Ages in Test set (86 null values)
# test_age_mean = full_data.Age.mean()
# test_age_std = full_data.Age.std()
# test_age_null = test_data.Age.isnull().sum()
# rand_ts_age = np.random.randint(test_age_mean - test_age_std, test_age_mean + test_age_std, size=test_age_null)
# test_data['Age'][np.isnan(test_data['Age'])] = rand_ts_age
# test_data['Age'] = test_data['Age'].astype(int)
# 
# train_data.Age = (train_data.Age/15).astype(np.int64)
# test_data.Age = (test_data.Age/15).astype(np.int64) + 1

# rint(train_data[['Age', 'Survived']].groupby(['Age'], as_index = False).mean())

# # SibSp and Parch

# In[ ]:


train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
fx, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].set_title('Family Size counts')
axes[1].set_title('Survival Rate vs Family Size')
fig1_family = sns.countplot(x=train_data.FamilySize, ax=axes[0], palette='cool')
fig2_family = sns.barplot(x=train_data.FamilySize, y=train_data.Survived, ax=axes[1], palette='cool')


# In[ ]:


train_data['Survived'].groupby(train_data['FamilySize']).mean()


# # isAlone

# In[ ]:


train_data['isAlone'] = train_data['FamilySize'].map(lambda x: 1 if x == 1 else 0)
test_data['isAlone'] = test_data['FamilySize'].map(lambda x: 1 if x == 1 else 0)


# In[ ]:


fx, axes = plt.subplots(1, 2, figsize=(15, 6))
fig1_alone = sns.countplot(data=train_data, x='isAlone', ax=axes[0])
fig2_alone = sns.barplot(data=train_data, x='isAlone', y='Survived', ax=axes[1])


# In[ ]:


train_data.drop(labels=['SibSp', 'Parch'], axis=1, inplace=True)
test_data.drop(labels=['SibSp', 'Parch'], axis=1, inplace=True)
train_data.head()


# # Ticket

# In[ ]:


train_data['Ticket_Len'] = train_data['Ticket'].apply(lambda x: len(x))
test_data['Ticket_Len'] = test_data['Ticket'].apply(lambda x: len(x))


# In[ ]:


fx, axes = plt.subplots(2, 1, figsize=(15, 10))
axes[0].set_title("Ticket Length vs Frequency")
axes[1].set_title("Length vise Survival rate")
fig1_tlen = sns.countplot(data=train_data, x='Ticket_Len', ax=axes[0])
fig2_tlen = sns.barplot(data=train_data, x='Ticket_Len',y='Survived', ax=axes[1])


# In[ ]:



train_data['Survived'].groupby(train_data['Ticket_Len']).mean()


# In[ ]:


train_data.drop(labels='Ticket', axis=1, inplace=True)
test_data.drop(labels='Ticket', axis=1, inplace=True)
train_data.head()


# # Fare

# In[ ]:


test_data['Fare'][np.isnan(test_data['Fare'])] = test_data.Fare.mean()


# In[ ]:


fx, axes = plt.subplots(1, 2, figsize=(15,5))
fig1_fare = sns.distplot(a=train_data.Fare, bins=15, ax=axes[0], hist_kws={'rwidth':0.7})
fig1_fare.set_title('Fare vise Frequency')

# Creating a new list of survived and dead

pass_survived_fare = train_data[train_data.Survived == 1].Fare
pass_dead_fare = train_data[train_data.Survived == 0].Fare

axes[1].hist(x=[train_data.Fare, pass_survived_fare, pass_dead_fare], bins=5, label=['Total', 'Survived', 'Dead'],         log=True)
axes[1].legend()
axes[1].set_title('Fare vise Survival')
plt.show()


# In[ ]:


train_data.Fare = (train_data.Fare/20).astype(np.int64) + 1 
test_data.Fare = (test_data.Fare/20).astype(np.int64) + 1


# In[ ]:


#print(training_data[['Fare','Survived']].groupby(['Fare'], as_index = False).mean())
train_data['Survived'].groupby(train_data['Fare']).mean()


# In[ ]:


train_data.head()


# # Cabin

# In[ ]:


cabin_null = float(test_data.Cabin.isnull().sum())
print(cabin_null/len(test_data) *100)


# In[ ]:


cabin_null = float(train_data.Cabin.isnull().sum())
print(cabin_null/len(train_data) *100)


# In[ ]:


train_data['hasCabin'] = train_data.Cabin.notnull().astype(int)
test_data['hasCabin'] = test_data.Cabin.notnull().astype(int)


# In[ ]:


fx, axes = plt.subplots(1, 2, figsize=(15, 6))
fig1_hascabin = sns.countplot(data=train_data, x='hasCabin', ax=axes[0])
fig2_hascabin = sns.barplot(data=train_data, x='hasCabin', y='Survived', ax=axes[1])


# In[ ]:


train_data.drop(labels='Cabin', axis=1, inplace=True)
train_data.head()


# In[ ]:


test_data.drop(labels='Cabin', axis=1, inplace=True)
test_data.head()


# # Embarked

# In[ ]:


train_data['Embarked'].describe()


# In[ ]:


train_data['Embarked'] = train_data['Embarked'].fillna('S')


# In[ ]:


fx, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].set_title('Embarked Counts')
axes[1].set_title('Survival Rate vs Embarked')
fig1_embarked = sns.countplot(x=train_data.Embarked, ax=axes[0])
fig2_embarked = sns.barplot(x=train_data.Embarked, y=train_data.Survived, ax=axes[1])


# In[ ]:



train_data['Survived'].groupby(train_data['Embarked']).mean()


# In[ ]:


#print(training_data[['Embarked', 'Fare']].groupby(['Embarked'], as_index = False).mean())
train_data['Fare'].groupby(train_data['Embarked']).mean()


# In[ ]:


ports = {'S': 0, 'C': 1, 'Q':2}
data = [train_data, test_data]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)


# # Cleaning Data

# In[ ]:


features = ['Pclass', 'Age', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'hasCabin']
train_data.head()


# In[ ]:


train_data[features].info()


# In[ ]:


train_data[features].head()


# In[ ]:



X = train_data[features]
y = train_data['Survived']
X_test = test_data[features]
X.head()


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
X_train, val_X, y_train, val_y = train_test_split(X, y, random_state=0)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rdmf = RandomForestClassifier(n_estimators=100, max_depth = 10, random_state = 1)
rdmf.fit(X_train, y_train)
val_predictions = rdmf.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# In[ ]:


predictions = rdmf.predict(X_test)
output = pd.DataFrame({'Survived': predictions})
output.to_csv('my_submission4.csv', index=False)


# In[ ]:




