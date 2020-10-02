#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.describe()


# In[ ]:


train.head()


# In[ ]:


# data cleaning
# drop name and id column
toDropCols = ['Name', 'PassengerId']
train.drop(toDropCols, axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


# drop ticket column and cabin because their values
toDropCols = ['Ticket', 'Cabin']
train.drop(toDropCols, axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


# drop more columns so that we could use linear regression with one variable
moreToDropCols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare' , 'Embarked']
ageTrain = train.drop(moreToDropCols, axis=1)


# In[ ]:


ageTrain.head()


# In[ ]:


train.head()


# In[ ]:


test.describe()


# In[ ]:


test.head()


# In[ ]:


# drop unnecessary columns in test accordingly
dropping = ['PassengerId', 'Name', 'Ticket', 'Cabin']
test.drop(dropping,axis=1, inplace=True)


# In[ ]:


test.head()


# In[ ]:


# selecting features
# get_dummies function
def getDummies(col,train,test):
    train_dum = pd.get_dummies(train[col])
    test_dum = pd.get_dummies(test[col])
    train = pd.concat([train, train_dum], axis=1)
    test = pd.concat([test,test_dum],axis=1)
    train.drop(col,axis=1,inplace=True)
    test.drop(col,axis=1,inplace=True)
    return train, test


# In[ ]:


# Pclass
# list Pclass values distribution and ensure that no na is included
print(train.Pclass.value_counts(dropna=False))


# In[ ]:


import seaborn as sns


# In[ ]:


sns.factorplot('Pclass', 'Survived', data=train, order=[1,2,3])


# In[ ]:


# keep this feature based on the difference from different Pclass
train, test = getDummies('Pclass', train, test)


# In[ ]:


# Sex
print(train.Sex.value_counts(dropna=False))
sns.factorplot('Sex','Survived', data=train)


# In[ ]:


# female survival rate is way better than the male
train, test = getDummies('Sex', train, test)
# drop the male column because it is not important from the graph
train.drop('male',axis=1,inplace=True)
test.drop('male',axis=1,inplace=True)


# In[ ]:


# Age
print(train.Age.value_counts(dropna=False))


# In[ ]:


import numpy as np


# In[ ]:


# data processing: 177 NaN missing values
missing_num = 177
# fill with random int that distributes around the mean with noise
age_mean = train['Age'].mean()
age_std = train['Age'].std()
filling = np.random.randint(age_mean-age_std, age_mean+age_std, size=missing_num)
train['Age'][train['Age'].isnull()==True] = filling
# check filling result
train['Age'].isnull().sum()


# In[ ]:


# process test data as well
missing_num_test = test['Age'].isnull().sum()
age_mean = test['Age'].mean()
age_std = test['Age'].std()
filling = np.random.randint(age_mean-age_std,age_mean+age_std,size=missing_num_test)
test['Age'][test['Age'].isnull()==True]=filling
missing_num_test = test['Age'].isnull().sum()
missing_num_test


# In[ ]:


#look into the age col
s = sns.FacetGrid(train,hue='Survived',aspect=3)
s.map(sns.kdeplot,'Age',shade=True)
s.set(xlim=(0,train['Age'].max()))
s.add_legend()


# In[ ]:


# from the graph, we see that the survival rate of children
# is higher than others and the 15-30 survival rate is lower
def under15(row):
    result = 0.0
    if row<15:
        result = 1.0
    return result
def young(row):
    result = 0.0
    if row>=15 and row<30:
        result = 1.0
    return result

train['under15'] = train['Age'].apply(under15)
test['under15'] = test['Age'].apply(under15)
train['young'] = train['Age'].apply(young)
test['young'] = test['Age'].apply(young)

train.drop('Age',axis=1,inplace=True)
test.drop('Age',axis=1,inplace=True)


# In[ ]:


# Family
# check
print(train['SibSp'].value_counts(dropna=False))
print(train['Parch'].value_counts(dropna=False))

sns.factorplot('SibSp','Survived',data=train,size=5)
sns.factorplot('Parch','Survived',data=train,size=5)


# In[ ]:


# both suggest we could sum family members as a feature
train['family'] = train['SibSp'] + train['Parch']
test['family'] = test['SibSp'] + test['Parch']
sns.factorplot('family','Survived',data=train,size=5)


# In[ ]:


# drop columns accordingly
train.drop(['SibSp','Parch'],axis=1,inplace=True)
test.drop(['SibSp','Parch'],axis=1,inplace=True)


# In[ ]:


# Fare
# checking null, found one in test group. leave it alone til we find out
# wether we should use this ft
train.Fare.isnull().sum()
test.Fare.isnull().sum()

sns.factorplot('Survived','Fare',data=train,size=5)


# In[ ]:


# keep this feature and fill up missing values
test['Fare'].fillna(test['Fare'].median(),inplace=True)


# In[ ]:


# Embark
train.Embarked.isnull().sum()
train.Embarked.value_counts()
# fill the majority val,'s', into missing val col
train['Embarked'].fillna('S',inplace=True)

sns.factorplot('Embarked','Survived',data=train,size=6)

