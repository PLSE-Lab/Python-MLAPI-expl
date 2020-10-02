#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn 


# In[ ]:


df_train=pd.read_csv("../input/titanic/train.csv")
df_test=pd.read_csv("../input/titanic/test.csv")


# In[ ]:


df_train.head(10)
df_test.head(10)


# In[ ]:


df_train.isnull().sum()
df_test.isnull().sum()


# In[ ]:


df_train['Embarked'].value_counts()


# In[ ]:


sns.countplot(x='Survived', data=df_train)


# In[ ]:


df_train.Embarked.unique()
df = df_train.groupby('Embarked').count()
sns.countplot(x='Survived',hue='Embarked', data=df_train)


# In[ ]:


df_train['Embarked'] = df_train['Embarked'].fillna('S')


# In[ ]:


sns.distplot(df_train['Age'].dropna(),kde=False)


# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df_train,palette='winter')


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[ ]:


df_train['Age'] = df_train[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df_test,palette='winter')


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 42

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[ ]:


df_test['Age'] = df_test[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


df_test['Fare'].fillna((df_test['Fare'].mean()), inplace=True)


# In[ ]:


def clean_table(x):
    try:
        return x[0]
    except TypeError:
        return "None"
    
    
df_train['Cabin']= df_train.Cabin.apply(clean_table)
df_test['Cabin']= df_test.Cabin.apply(clean_table)


# In[ ]:


catagorical_variables=['Sex','Cabin','Embarked']

for i in catagorical_variables:
    df_train[i].fillna("Missing", inplace=True)
    dummies=pd.get_dummies(df_train[i], prefix=i)
    df_train=pd.concat([df_train,dummies],axis=1)
    df_train.drop([i],axis=1,inplace=True)


# In[ ]:


for i in catagorical_variables:
    df_test[i].fillna("Missing", inplace=True)
    dummies=pd.get_dummies(df_test[i], prefix=i)
    df_test=pd.concat([df_test,dummies],axis=1)
    df_test.drop([i],axis=1,inplace=True)


# In[ ]:


df_train.drop(['Name','PassengerId'],axis=1,inplace=True)
df_test.drop(['Name','PassengerId'],axis=1,inplace=True)
df_train.drop(['Ticket'],axis=1,inplace=True)
df_test.drop(['Ticket'],axis=1,inplace=True)


# In[ ]:


X_train=df_train.drop(['Survived'],axis=1)
y_train=df_train['Survived']
X_train.drop('Cabin_T', axis=1,inplace=True)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
Classifier = RandomForestClassifier(n_estimators=20, criterion='entropy')


# In[ ]:


Classifier.fit(X_train,y_train)


# In[ ]:


y_pred=Classifier.predict(df_test)
y_pred


# In[ ]:


sub_df=pd.read_csv('../input/titanic/gender_submission.csv')
submission={}
submission['PassengerId']= sub_df.PassengerId
submission['Survived']= y_pred
submission=pd.DataFrame(submission)
submission.to_csv('mysubmission5.csv',index=False)

