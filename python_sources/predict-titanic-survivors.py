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
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# reading datasets using pandas
data_train = pd.read_csv('../input/titanic/train.csv')
data_test = pd.read_csv('../input/titanic/test.csv')
data_train.head()


# We have a peek on our data and want to know deeper about our data.

# In[ ]:


data_train.info()


# So there are 891 entries in our data with 12 columns. 
# There are categorical datas:
# 1. PassengerId : ID of passenger
# 2. Survived : whether the passenger survived from the crash or not
# 3. Pclass : ticket class of passenger
# 4. Name : name of the passenger
# 5. Sex : whether passenger is male or female
# 6. Ticket : ticket number
# 7. Cabin : cabin number
# 8. Embarked : where the passenger embarked from
# 
# And also numerical datas:
# 1. Age : age of passenger
# 2. Sibsp : number of siblings or spouse that the passenger went aboard together
# 3. Parch : number of parent or children that the passenger went aboard together
# 4. Fare : price of the ticket
# 
# We will make machine learning to predict if a passenger will survive or not from disaster, so our target is 'survived' column. Now we will do more EDA to choose which features that we will use to train our model.

# ### Exploratory Data Analysis

# In[ ]:


sns.set_style('darkgrid')
ax = sns.countplot(data_train['Survived'])
for p in ax.patches:
        ax.annotate('{} ({:.1f}%)'.format(p.get_height(), 100* p.get_height()/len(data_train['Survived'])), 
                    (p.get_x()+0.2, p.get_height()-30))


# Passengers who didn't survived were almost twice as those who survived.<br>
# We will see if some aspect like sex, Pclass, embarked port have influence in this odd.

# In[ ]:


plt.figure(figsize=(12,6))

plt.subplot(121)
ax = sns.countplot(data = data_train, x='Sex')
for p in ax.patches:
        ax.annotate('{} ({:.1f}%)'.format(p.get_height(), 100* p.get_height()/len(data_train['Sex'])), 
                    (p.get_x()+0.2, p.get_height()-30))

plt.subplot(122)
ax = sns.countplot(data = data_train, x='Survived', hue='Sex')
for p in ax.patches:
        ax.annotate('{} ({:.1f}%)'.format(p.get_height(), 100* p.get_height()/len(data_train['Survived'])), 
                    (p.get_x()+0.01, p.get_height()-30))
        
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data = data_train, x='Survived', hue='Pclass')
for p in ax.patches:
        ax.annotate('{} ({:.1f}%)'.format(p.get_height(), 100* p.get_height()/len(data_train['Survived'])), 
                    (p.get_x()+0.03, p.get_height()-30))


# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data = data_train, x='Survived', hue='Embarked')
for p in ax.patches:
        ax.annotate('{} ({:.1f}%)'.format(p.get_height(), 100* p.get_height()/len(data_train['Survived'])), 
                    (p.get_x()+0.03, p.get_height()-25))


# Some remarks that I note:
# 1. 52.5% of passenger who didn't survived were male, there 64.8% male passenger in the ship, most of the male didn't survived.
# 2. 41.8% of passenger who didn't survived were 3rd class passenger 
# 
# We can gain some more insight from the data if we like. Some examples are below.

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data = data_train, x='Sex', hue='Pclass')
for p in ax.patches:
        ax.annotate('{} ({:.1f}%)'.format(p.get_height(), 100* p.get_height()/len(data_train['Survived'])), 
                    (p.get_x()+0.03, p.get_height()-30))


# We have seen before that female has more likelihood to survive than male. Higher class passenger also has more survival number than lower class. From the graphic above we can see that there are lots of male passenger in 3rd class. Maybe most of passengers who didn't survived come from this category. 

# In[ ]:


plt.figure(figsize=(12,6))
ax = sns.countplot(data = data_train, x='Embarked', hue='Pclass')
for p in ax.patches:
        ax.annotate('{} ({:.1f}%)'.format(p.get_height(), 100* p.get_height()/len(data_train['Survived'])), 
                    (p.get_x()+0.01, p.get_height()+5))


# There were 3 ports where passenger can embark
# 1. S - Southampton, England
# 2. C - Cherbourg, France
# 3. Q - Queenstown, Ireland
# 
# Most passengers were come from England.

# We continue our EDA by looking some numerical values.

# In[ ]:


data_train.drop(['PassengerId', 'Survived', 'Pclass'], axis=1).describe()


# There are 177 entries out of 891 in age column that are missing. We need to deal with it late.
# <br> Age of passengers was mostly around 20 - 40 years old.
# <br> There were not so many passengers who went on board with their relatives i.e parents, children, siblings, spouse.
# <br> Range of fare ticket were vary from free ticket like 0 to very high price like 512. It can be happen that there are some outliers in this field.
# <br> We will see it more clearly using graphic visualization below.

# In[ ]:


sns.distplot(data_train['Age'].dropna(), kde=False)


# In[ ]:


plt.figure(figsize=(10,6))
plt.subplot(121)
sns.countplot(data=data_train, x='SibSp')
plt.subplot(122)
sns.countplot(data=data_train, x='Parch')
plt.tight_layout()


# In[ ]:


sns.distplot(data_train['Fare'])


# ### Data Preprocessing

# Now we prepare our data before we use it to train our model. 
# First we will deal with missing value.

# In[ ]:


data_train.isnull().sum()


# Since there are too much missing value in 'cabin' column, I choose to drop this column for our model. I using mode to fill missing value in embarked column.

# In[ ]:


figure = plt.figure(figsize=(10,6))
sns.boxplot(data=data_train, x='Pclass', y='Age')


# Based off above graphic there is tendency that older people can afford to buy higher class ticket. So I will use mean of age in each passenger class to fill the missing value in age column.  

# In[ ]:


first_mean = round(data_train[data_train['Pclass'] == 1]['Age'].dropna().mean())
second_mean = round(data_train[data_train['Pclass'] == 2]['Age'].dropna().mean())
third_mean = round(data_train[data_train['Pclass'] == 3]['Age'].dropna().mean())

# creating function to fill missing age
def filling(col):
    Age = col[0]
    Pclass = col[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return first_mean
        elif Pclass == 2:
            return second_mean
        else:
            return third_mean
    else:
        return Age

data_train['Age'] = data_train[['Age', 'Pclass']].apply(filling, axis=1)


# In[ ]:


data_train['Embarked'].mode()


# In[ ]:


data_train['Embarked'].fillna('S', inplace=True)
data_train.isnull().sum()


# Now to deal with categorical data (sex and embarked), I use dummy variable.

# In[ ]:


sex = pd.get_dummies(data_train['Sex'],drop_first=True)
embarked = pd.get_dummies(data_train['Embarked'],drop_first=True)


# Finally we have our training dataset.

# In[ ]:


X = pd.concat([data_train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']], sex, embarked], axis=1) #creating our features
y= data_train['Survived'] #chosing our target
print(X.head())
print(y.head())


# Let's take a peek on our test set.

# In[ ]:


data_test.head()


# In[ ]:


data_test.info()


# In[ ]:


data_test.isnull().sum()


# In[ ]:


data_test.drop(['PassengerId', 'Pclass'], axis=1).describe()


# It looks like our test data has quite same behaviour with our train data. Lots of missing value in cabin column. Great variance in fare price. Also some missing value in age column. I think we can treat it the same afterward.

# In[ ]:


data_test['Fare'].fillna('35.62', inplace=True)
data_test['Age'] = data_test[['Age', 'Pclass']].apply(filling, axis=1)
sex = pd.get_dummies(data_test['Sex'],drop_first=True)
embarked = pd.get_dummies(data_test['Embarked'],drop_first=True)
X_validation = pd.concat([data_test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']], sex, embarked], axis=1)
X_validation.head()


# ### Creating machine learning model
# 
# One of common algorithm for classification problem is Random Forest Classification. I will use that algorithm to create my model.

# To prevent the model from overfit. I will split our training data. So I will have
# 1. Train set
# 2. Test set
# 3. Validation set -> data test provided for prediction submission

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=46)

from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

y_predict = model.predict(X_test)
print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))


# In[ ]:


predictions = model.predict(X_validation)
data_test['Survived'] = predictions
submit = data_test[['PassengerId', 'Survived']]
submit.to_csv('submission.csv', index=False)

