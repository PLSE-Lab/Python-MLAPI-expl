
# coding: utf-8

# In[455]:


import os
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn import linear_model
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[456]:

test.head()


# In[457]:

def Title(df):
    Title = []
    for i in df['Name']:
        b1 = i.split('. ')
        b2 = b1[0].split(', ')
        Title.append(b2[1])    

    Title1 = []
    for i in Title:
        if (i == 'Ms' or i == 'Lady' or i == 'Mrs'):
            i = 'Mrs'
            Title1.append(i)
        elif (i == 'Mlle' or i == 'Mme' or i == 'Miss'):
            i = 'Miss'
            Title1.append(i)
        elif i == 'Master':
            i = 'Master'
            Title1.append(i)
        elif i == 'Mr':
            i = 'Mr'
            Title1.append(i)      
        else:
            i = 'Others'
            Title1.append(i)  
    df['Title'] = Title1   
    return df


# In[458]:

train = Title(train)
test = Title(test)


# In[459]:

def clean_df(df):
    df['Age'] = df['Age'].fillna(df.Age.median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Family'] = df['Parch'] + df['SibSp']
    df['Fare'] = df['Fare'].fillna(df.Fare.median())
    df['Gender'] = df['Sex'].map({'female':0,'male':1}).astype(int)
    df['Title'] = df['Title'].map({'Others':0, 'Master':1, 'Mrs':2, 'Miss':3, 'Mr':4}).astype(int)
    clean_df = df.drop(['Name','Sex','Ticket','Cabin','PassengerId','SibSp', 'Parch'],axis=1)
    return clean_df
cleaned_train = clean_df(train)
cleaned_test = clean_df(test)
cleaned_test.head()


# In[460]:

train.head()


# In[461]:

Title = set()
for i in train['Title']:
    Title.add(i)
Title = list(Title)
    
#for i in Title:
 #   print i, 'Survived :', sum((train['Title'] == i) & train['Survived'] == 1)
  #  print i, 'Died :', sum(train['Title'] == i) - sum((train['Title'] == i) & train['Survived'] == 1)


# In[462]:

sns.set_style('whitegrid')
sns.factorplot(x = 'Embarked',y = 'Survived', data = train,size = 4, aspect = 3)
#sns.factorplot('Embarked','Survived', data=train,size=4,aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
sns.countplot('Embarked', data = train, ax = axis1)
sns.countplot(x = 'Survived',hue = 'Embarked',data = train, ax = axis2)
embark_perc = train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x = 'Embarked', y = 'Survived', data = embark_perc,order = ['S', 'C', 'Q'], ax = axis3)


# In[463]:

fig = plt.figure(figsize=(18, 8))
ax1 = plt.subplot2grid((2,3), (0,0))
train.Age.value_counts().plot(kind='kde', color='#FA2379', label='train', alpha=0.6)

ax2 = plt.subplot2grid((2,3), (0,1))
train.Pclass.value_counts().plot(kind='barh', color='#FA2379', label='train', alpha=0.6)

ax3 = plt.subplot2grid((2,3), (0,2))
train.Sex.value_counts().plot(kind='barh', color='#FA2379', label='train', alpha=0.6)

ax4 = plt.subplot2grid((2,3), (1,0), colspan=2)
train.Fare.value_counts().plot(kind='kde', color='#FA2379', label='train', alpha=0.6)

ax5 = plt.subplot2grid((2,3), (1,2))
train.Embarked.value_counts().plot(kind='barh', color='#FA2379', label='train', alpha=0.6)
train['Title'] = train['Title'].fillna('Mr')


# In[464]:

logistic = linear_model.LogisticRegression()
X = cleaned_train.drop(['Survived','Embarked'],axis=1)
y = train['Survived']
logistic.fit(X,y)
logistic.score(X, y)


# In[465]:

X_test = cleaned_test.drop(['Embarked'], axis = 1)
y_pred = logistic.predict(X_test)


# In[466]:

random_forest = ensemble.RandomForestClassifier(n_estimators=100)
random_forest.fit(X,y)
y_pred = random_forest.predict(X_test)
random_forest.score(X,y)


# In[468]:

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':y_pred})
submission.to_csv('Titanic_Submission.csv', index = False)


# In[ ]:



