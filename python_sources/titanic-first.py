#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os


# In[ ]:


os.listdir("../input/titanic")


# In[ ]:


train_data = pd.read_csv("../input/titanic/train.csv")


# In[ ]:


train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


sns.set_style(style='whitegrid')


# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=train_data);


# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=train_data);


# In[ ]:


sns.distplot(train_data['Age'].dropna(),kde=False,bins=30);


# In[ ]:


sns.countplot(train_data['SibSp'])


# In[ ]:


train_data.isnull().sum()


# In[ ]:


#Analysing the age and passenger class through boxplot
plt.figure(figsize=(10,8))
sns.boxplot(x='Pclass',y='Age',data=train_data)


# Passengers in the 1st class are older than the passengers in 3rd class except some outliers. This makes sense because, age is directly propotional to the wealth acquired throughtout the days.

# In[ ]:


def add_age(col):
    Age = col[0]
    Pclass = col[1]
    if pd.isnull(Age):
        #fill the null columns with the average value from the boxplot
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


train_data['Age'] = train_data[['Age','Pclass']].apply(add_age,axis=1)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


train_data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train_data.dropna(inplace=True)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


train_data.head()


# In[ ]:


sex = pd.get_dummies(train_data['Sex'],drop_first=True)


# In[ ]:


sex.head()


# In[ ]:


embark = pd.get_dummies(train_data['Embarked'],drop_first=True)


# In[ ]:


embark.head()


# In[ ]:


train_data = pd.concat([train_data,sex,embark],axis=1)


# In[ ]:


train_data.head()


# In[ ]:


train_data.columns


# In[ ]:


train_data = train_data.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1)


# In[ ]:


train_data.head()


# Test data

# In[ ]:


test = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


test_data = test.copy()


# In[ ]:


test_data.head()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(x='Pclass',y='Age',data=test_data)


# In[ ]:


def add_age_test(col):
    Age = col[0]
    Pclass = col[1]
    if pd.isnull(Age):
        #fill the null columns with the average value from the boxplot
        if Pclass == 1:
            return 42
        elif Pclass == 2:
            return 26
        else:
            return 24
    else:
        return Age


# In[ ]:


test_data['Age'] = test_data[['Age','Pclass']].apply(add_age_test,axis=1)


# In[ ]:


test_data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


test_data['Fare'].mean()


# In[ ]:


test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_data.head(155)


# In[ ]:


sex_test = pd.get_dummies(test_data['Sex'],drop_first=True)


# In[ ]:


sex_test.head()


# In[ ]:


embark_test = pd.get_dummies(test_data['Embarked'],drop_first=True)


# In[ ]:


embark_test.head()


# In[ ]:


test_data = pd.concat([test_data,sex_test,embark_test],axis=1)


# In[ ]:


test_data.head()


# In[ ]:


test_data = test_data.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1)


# In[ ]:


test_data.head()


# In[ ]:


X_train = train_data.drop(['Survived'],axis=1)


# In[ ]:


y_train = train_data['Survived'].values


# In[ ]:


X_test = test_data


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model = LogisticRegression()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


predictions = model.predict(X_test)


# In[ ]:


predictions.shape


# In[ ]:


real_values = pd.read_csv("../input/titanic/gender_submission.csv")


# In[ ]:


real_values.shape


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


confusion_matrix(real_values['Survived'].values,predictions)


# In[ ]:


print(classification_report(real_values['Survived'].values,predictions))


# In[ ]:


result = pd.DataFrame(predictions,columns=['Survived'])


# In[ ]:


result = pd.concat([test['PassengerId'],result],axis=1)


# In[ ]:


result


# In[ ]:


result.to_csv("submission.csv",index=False)


# In[ ]:


os.listdir("../working")

