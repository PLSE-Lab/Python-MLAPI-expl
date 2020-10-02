#!/usr/bin/env python
# coding: utf-8

# 
# # happy new year!
# 
# ** i wish you all the best in the new year **
# 
# this kernel is the beginning of the titanic to practice the kaggle like me
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


# Any results you write to the current directory are saved as output.


# In[ ]:


print(os.listdir("../input"))


# ** loading dataset **

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ** check data **

# In[ ]:


train.head()


# ** data description **
# 
# - survived : 0 = no, 1 = yes
# - Pclass : ticket class. 1 = 1st, 2 = 2nd..
# - sex : sex
# - sibsp : number of spouse
# - parch : number of children
# - ticket : ticket number
# - cabin : cabin number
# - embarked : a landed port

# data rows and columns
# 
# data information

# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.info()


# In[ ]:


train.describe()


# ** check null data **

# In[ ]:


train.isnull().sum()


# there is null data in Age, Cabin ( train data)

# In[ ]:


test.isnull().sum()


# and test also has null
# 

# ** check survived **

# In[ ]:


survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]


# In[ ]:


print("surv : %.1f" %(len(survived) / len(train) * 100))
print("not surv : %.1f " %(len(not_survived) / len(train) * 100))


# In[ ]:


plt.bar(['survive', 'not survive'], [len(survived), len(not_survived)])
plt.title("survival")
plt.show()


# In[ ]:


survived_number_by_sex = train[train['Survived']==1]['Sex'].value_counts()
not_survived_number_by_sex = train[train['Survived']==0]['Sex'].value_counts()


# In[ ]:


survived_number_by_sex


# In[ ]:


not_survived_number_by_sex


# In[ ]:


tmp = pd.DataFrame([survived_number_by_sex, not_survived_number_by_sex])


# In[ ]:


tmp.index = ['sur', 'notsur']


# In[ ]:


tmp.head()


# In[ ]:


plt.bar(['female', 'male'], [tmp['female']['sur'], tmp['male']['sur']])
plt.xlabel('sex')
plt.ylabel('number of survived')
plt.grid()
plt.show()


# In[ ]:


sns.barplot(x='Sex', y='Survived', data=train)


# ** we can see that more women survived than men **

# In[ ]:


number_of_pclass = train['Pclass'].value_counts()
number_of_pclass


# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train)


# ** and you can see survival rate by Pclass **

# 

# In[ ]:


sns.factorplot('Pclass', 'Survived', hue='Sex', data=train)
plt.show()


# you can see that there are more women suviving then men
# 
# the lower the pclass, more perple died
# 

# # lets try age -> continous feature

# In[ ]:


print(train['Age'].max())
print(train['Age'].min())
print(train['Age'].mean())


# In[ ]:


f, ax = plt.subplots(figsize=(12,8))
sns.violinplot("Pclass", "Age", hue="Survived", data=train, split=True)
ax.set_yticks(range(0, int(train['Age'].max())+10, 10))
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(12,8))
sns.violinplot("Sex", "Age", hue='Survived', split=True, data=train)
ax.set_yticks(range(0, int(train['Age'].max() + 10), 10))
plt.show()


# in the pclass ratio, many children under 10 survived.
# 
# survival chances for passengers aged 20-50 from Pclass1 is high and is even better for women.

# ** as we saw above, there are quite a few null values in age **
# 
# it is too different for the average value. so we access to this problem from another way.
# 
# another easy to access method is to use the name feature!
# 

# ** using a regular expression, calculate the value of Mr. and Mrs. etc **

# In[ ]:


import re
p = re.compile('([A-Za-z]+)\.')
for cnt, value in enumerate(train['Name']):
    print(p.search(value).group())
    if cnt >= 5: break


# as you can see there are Mr, Mrs, Miss etc in name.

# In[ ]:


train['Initial'] = ''
init = []
p = re.compile('([A-Za-z]+)\.')
for cnt, value in enumerate(train['Name']):
    init.append(p.search(value).group())
train['Initial'] = init


# In[ ]:


train['Initial'].head(10)


# In[ ]:


train['Initial'].unique()


# In[ ]:


train['Initial'].value_counts()


# there are Mr, Mrs, Miss, Master, Don etc.....

# In[ ]:


pd.crosstab(train.Initial, train.Sex).T.style.background_gradient(cmap='winter')


# In[ ]:


pre = ['Mr.', 'Miss.', 'Mrs.', 'Master.', 'Mlle.','Mme.','Ms.','Dr.','Major.','Lady.','Countess.','Jonkheer.','Col.','Rev.','Capt.','Sir.','Don.']
aft = ['Mr', 'Miss', 'Mrs', 'Master', 'Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr']
train['Initial'].replace(pre,aft,inplace=True)


# In[ ]:


train['Initial'].value_counts()


# In[ ]:


train.groupby('Initial')['Age'].head(1)


# In[ ]:


train.groupby('Initial')['Age'].mean()


# ** we extract the average of each Initial feature **

#  # filling NaN ages

# In[ ]:


train.loc[(train.Age.isnull()) & (train.Initial == 'Master'), 'Age'] = 5
train.loc[(train.Age.isnull()) & (train.Initial == 'Miss'), 'Age'] = 22
train.loc[(train.Age.isnull())&(train.Initial=='Mr'), 'Age'] = 33
train.loc[(train.Age.isnull()) & (train.Initial=='Mrs'), 'Age'] = 36
train.loc[(train.Age.isnull()) & (train.Initial == 'Other'), 'Age'] = 46


# In[ ]:


train['Age'].isnull().sum()


# In[ ]:


f, ax = plt.subplots(1, 2, figsize=(18,10))
train[train['Survived'] == 0]['Age'].plot.hist(ax=ax[0], bins=20, edgecolor='black', color='red')
ax[0].set_title('not survived')
x = list(range(0, 90, 5))
ax[0].set_xticks(x)

train[train['Survived'] == 1]['Age'].plot.hist(ax=ax[1], bins=20, edgecolor='black', color='blue')
ax[1].set_title('survived')
ax[1].set_xticks(x)

plt.show()


# we can see that -> 
# 
# babies were saved in large number
# 
# maxinum number of deaths were in the age group of 30-40

# In[ ]:


sns.factorplot('Pclass', 'Survived', col='Initial', data=train)
plt.show()


# ** now we check the embarked feautre **

# In[ ]:


pd.crosstab([train['Embarked'], train['Pclass']], [train['Sex'], train['Survived']]).style.background_gradient(cmap='winter')


# In[ ]:


sns.factorplot('Embarked', 'Survived', data=train)
plt.show()


# the port C is highest around 0.55 whie it is lowest for S

# In[ ]:


f, ax = plt.subplots(2,2, figsize=(18,15))
sns.countplot('Embarked', data=train, ax=ax[0,0])
ax[0,0].set_title('number of boarded ')

sns.countplot('Embarked', data=train, hue='Sex', ax=ax[0,1])

sns.countplot('Embarked', data=train, hue='Survived', ax=ax[1,0])

sns.countplot('Embarked', data=train, hue='Pclass', ax=ax[1,1])
plt.show()


# ** we can see that **
# 
# 1. many passengers got on board at S.
# 2. Survival numbers are higher than the probability of not surviving in C only.
# 3. S was ridden by rich people. however, chances of survival are low. because there were many pclass3 passengers too and most of plcass3 passengers did not survived.

# In[ ]:


sns.factorplot('Pclass', 'Survived', hue='Sex', col='Embarked', data=train)
plt.show()


# survival chances are almost 1 for women for Pclass1, Pclass2

# as we canss see above, there is also NaN value in Embark features.
# 
# maximum passengers boarded from port S. so replcace NaN with S.

# In[ ]:


train['Embarked'].fillna('S', inplace=True)


# In[ ]:


train['Embarked'].isnull().sum()


# ** check sibsip vs survived **
# 
# sibling = brother, sister ect
# 
# spouse = husband, wife

# In[ ]:


train['SibSp'].value_counts()


# In[ ]:


train.groupby('SibSp')['Survived'].value_counts()


# In[ ]:


pd.crosstab([train['SibSp']], train['Survived']).style.background_gradient(cmap='winter')


# In[ ]:


sns.barplot(x='SibSp', y='Survived', data=train)


# ** check Parch feature **
# 
# 

# In[ ]:


train['Parch'].value_counts()


# In[ ]:


pd.crosstab(train['Parch'], train['Survived']).style.background_gradient(cmap='winter')


# In[ ]:


sns.barplot(x='Parch', y='Survived', data=train)


# we can see that :
# 
# having 1-2 siblings shows a greater chance of propablity rather than being alone and having a large family.
# 
# 
# 
# ** check fare **

# In[ ]:


print(train['Fare'].max())
print(train['Fare'].min())
print(train['Fare'].mean())


# In[ ]:


f, ax = plt.subplots(1,3 , figsize=(20,8))
sns.distplot(train[train['Pclass'] == 1]['Fare'], ax=ax[0])
ax[0].set_title('Fares in Pclass 1')

sns.distplot(train[train['Pclass'] == 2]['Fare'], ax=ax[1])
ax[1].set_title('Fares in Pclass 2')

sns.distplot(train[train['Pclass'] == 3]['Fare'], ax = ax[2])
ax[2].set_title('Fares in Pclass 3')

plt.show()


# # correlation between the features
# 

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(train.drop('PassengerId', axis=1).corr(), annot=True, linewidths=0.2, cmap='PuBu')


# ** a value 1 means perfect positive correlation. **
# 
# ** a value -1 means perfect negative correlation **

# # feature engineering
# 

# In[ ]:


del train


# In[ ]:


del test


# # reload data and cleaning
# 

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train_test = [train, test]


# In[ ]:


train_test[0].head()


# In[ ]:


print(train_test[0].shape)
print(train_test[1].shape)


# In[ ]:


for data in train_test:
    data['Initial'] = ''
    init = []
    p = re.compile('([A-Za-z]+)\.')
    for value in data['Name']:
        init.append(p.search(value).group())
    data['Initial'] = init


# In[ ]:


train.head()


# In[ ]:


pre = ['Mr.', 'Miss.', 'Mrs.', 'Master.', 'Mlle.','Mme.','Ms.','Dr.','Major.','Lady.','Countess.','Jonkheer.','Col.','Rev.','Capt.','Sir.','Don.', 'Dona.']
aft = ['Mr', 'Miss', 'Mrs', 'Master', 'Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Other']
for data in train_test:
    data['Initial'].replace(pre,aft,inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train['Initial'].value_counts()


# In[ ]:


train.groupby('Initial')['Age'].head(1)


# In[ ]:


train.groupby('Initial')['Age'].mean()


# In[ ]:


for data in train_test:
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Master'), 'Age'] = 5
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Miss'), 'Age'] = 22
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Mr'), 'Age'] = 33
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Mrs'), 'Age'] = 36
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Other'), 'Age'] = 46


# In[ ]:


train['Age'].isnull().sum()


# In[ ]:


test['Age'].isnull().sum()


# and convert the categorical 'Initial' values into numeric form

# In[ ]:


mapping = {
    'Mr' : 1,
    'Miss' : 2,
    'Mrs' : 3,
    'Master' : 4,
    'Other' : 5
}


# In[ ]:


for data in train_test:
    data['Initial'] = data['Initial'].map(mapping).astype(int)


# In[ ]:


train.head()


# In[ ]:


train['Initial'].value_counts()


# convert 'sex' feature to numerical

# In[ ]:


mapping ={
    'female' : 1,
    'male': 0
}


# In[ ]:


for data in train_test:
    data['Sex'] = data['Sex'].map(mapping).astype(int)


# In[ ]:


train['Sex'].value_counts()


# In[ ]:


train.head()


# filled 'Emabark' NaN values

# In[ ]:


for data in train_test:
    data['Embarked'].fillna('S', inplace=True)


# In[ ]:


print(train['Embarked'].unique())
print(test['Embarked'].unique())


# In[ ]:


mapping = {
    'S' : 0,
    'C' : 1,
    'Q' : 2
}


# In[ ]:


for data in train_test:
    data['Embarked'] = data['Embarked'].map(mapping).astype(int)


# ** chagne age band. **
# 
# age is continous feature. we need to convert these continuous values into categorical values by binning.

# In[ ]:


for data in train_test:
    data.loc[data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16 ) & ( data['Age'] <= 32 ), 'Age'] = 1
    data.loc[(data['Age'] > 32 ) & ( data['Age'] <= 48 ), 'Age'] = 2
    data.loc[(data['Age'] > 48 ) & ( data['Age'] <= 64 ), 'Age'] = 3
    data.loc[data['Age'] > 64, 'Age'] = 4
    


# In[ ]:


train.head()


# In[ ]:


print(train['Age'].unique())
print(test['Age'].unique())


# In[ ]:


train['Age'] = train['Age'].astype(int)
test['Age'] = test['Age'].astype(int)


# In[ ]:


print(train['Age'].unique())
print(test['Age'].unique())


# In[ ]:


train['Age'].value_counts().to_frame().style.background_gradient('summer')


# Fare feautre

# In[ ]:


print(train['Fare'].isnull().sum())
print(test['Fare'].isnull().sum())


# fill NaN values by train median

# In[ ]:


for data in train_test:
    data['Fare'].fillna(train['Fare'].median(), inplace=True)


# In[ ]:


print(test['Fare'].isnull().sum())


# In[ ]:


train['FareBand'] = pd.qcut(train['Fare'], 4) # four section.


# In[ ]:


train.groupby(['FareBand'])['Survived'].mean().to_frame().style.background_gradient('summer')


# to categorical
#     

# In[ ]:


for data in train_test:
    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & ( data['Fare'] <= 14.454 ), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454 ) & (data['Fare'] <= 31), 'Fare' ] = 2
    data.loc[data['Fare'] > 31, 'Fare'] = 3


# In[ ]:


train['Fare'] = train['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)


# In[ ]:


print(train['Fare'].unique())
print(test['Fare'].unique())


# next, SibSp &* Parch Feature.
# 
# im going to combine these two features into one.

# In[ ]:


for data in train_test:
    data['Family'] = data['SibSp'] + data['Parch']


# In[ ]:


train[['Family', 'Survived']].groupby(['Family']).mean()


# we can see that family size 1~3 has high survival rate.

# In[ ]:


for data in train_test:
    data.loc[data['Family'] == 0, 'Family'] = 0
    data.loc[(data['Family'] >= 1) & (data['Family'] < 4), 'Family'] = 1
    data.loc[(data['Family'] >= 4) & (data['Family'] < 7), 'Family'] = 2
    data.loc[(data['Family'] >= 7), 'Family'] = 3


# In[ ]:


train['Family'].unique()


# In[ ]:


train[['Family', 'Survived']].groupby(['Family']).mean()


# In[ ]:


test.head(2)


# # we drop unnecessary features
# 
# the features list : name, ticket, cabin(becaluse a lot of NaN values), passengerId etc

# In[ ]:


drop_list = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin']
train = train.drop(drop_list, axis=1)
test = test.drop(drop_list, axis=1)

train = train.drop(['PassengerId', 'FareBand'], axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# # next weeks..
# 
# i will do prediction. make model, check accuracy

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




