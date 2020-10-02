#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#identifying the missing values


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(train['Survived'], hue= train['Pclass'])


# In[ ]:


sns.distplot(train['Age'].dropna(), bins=30)


# In[ ]:


sns.countplot(x='SibSp', data=train)


# In[ ]:


train['Fare'].hist( bins=40, figsize=(8,4))


# In[ ]:


import cufflinks as cf


# In[ ]:


cf.go_offline()


# In[ ]:


train['Fare'].iplot(kind='hist',bins=50)


# In[ ]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age', data=train)


# In[ ]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols [1]
    
    if pd.isnull(Age):
        
        if Pclass==1:
            return 38
        elif Pclass==2:
            return 29
        else:
            return 25
    else:
        return Age


# In[ ]:


train['Age']= train[['Age', 'Pclass']].apply(impute_age, axis=1)


# In[ ]:


plt.figure(figsize=(10,7))
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[ ]:


train.drop('Cabin', axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


Sex= pd.get_dummies(train['Sex'],drop_first=True)
Embarked= pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)
train.drop(['PassengerId'], axis=1, inplace=True)


# In[ ]:


train= pd.concat((train,Sex,Embarked), axis=1)


# In[ ]:


train.head()


# In[ ]:


x= train.drop(['Survived'] , axis=1)
y= train['Survived']


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel= LogisticRegression()


# In[ ]:


logmodel.fit(x, y)


# In[ ]:


test=pd.read_csv('../input/test.csv')
test.info()
test.head()


# In[ ]:


Sex= pd.get_dummies(test['Sex'],drop_first=True)
Embarked= pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Name', 'Sex', 'Ticket', 'Embarked','Cabin'], axis=1, inplace=True)


# In[ ]:


test= pd.concat((test,Sex,Embarked), axis=1)


# In[ ]:


test.head()


# In[ ]:


sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[ ]:


sns.boxplot(x=test['Pclass'],y=test['Fare'])


# In[ ]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols [1]
    
    if pd.isnull(Age):
        
        if Pclass==1:
            return 40.91
        elif Pclass==2:
            return 28.77
        else:
            return 24.02
    else:
        return Age


# In[ ]:


test['Age']= test[['Age', 'Pclass']].apply(impute_age, axis=1)


# In[ ]:


test[test['Pclass']==3].describe()


# In[ ]:


test[test.isnull().any(axis=1)]


# In[ ]:


test[test['PassengerId']==1044]


# In[ ]:


means = test.groupby('Pclass')['Fare'].transform('mean')
test['Fare'] = test['Fare'].fillna(means)


# In[ ]:


predict=logmodel.predict(test.drop(['PassengerId'],axis=1))


# In[ ]:


submission=pd.read_csv('../input/gender_submission.csv')
submission['Survived']=predict
submission.to_csv('titanic_submission.csv',index=False)


# In[ ]:




