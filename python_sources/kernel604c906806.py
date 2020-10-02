#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


# will check if there is a null value in any of the columns


# In[ ]:


import seaborn as sns 
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


#missing value in age and cabin columns in train dataset


# In[ ]:


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


#missing values in age and cabin columns of test dataset 


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=train)


# In[ ]:


def fill_age(cols):
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


train['Age'] = train[['Age','Pclass']].apply(fill_age,axis=1)


# In[ ]:


test['Age'] = test[['Age','Pclass']].apply(fill_age,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:




sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# In[ ]:


train.drop('Cabin',axis=1,inplace=True)
train.head(2)


# In[ ]:


test.drop('Cabin',axis=1,inplace=True)
test.head(2)


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


#one data is missing in fare columns


# In[ ]:


test[test.Fare!=test.Fare]


# In[ ]:


imp_fare = test.loc[(test.Embarked == 'S') & (test.Pclass == 3), "Fare"].mean()
test.loc[test.Fare!=test.Fare , "Fare"] = round(imp_fare, 2)
test.loc[(test.Name == "Storey, Mr.Thomas"),:]


# In[ ]:


test.info()


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


test.dropna(inplace=True)


# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
sex


# In[ ]:


embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark.head()


# In[ ]:


sex1 = pd.get_dummies(test['Sex'],drop_first=True)
sex1.head()


# In[ ]:


embark1 = pd.get_dummies(test['Embarked'],drop_first=True)
embark1.head()


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head(2)


# In[ ]:


test = pd.concat([test,sex1,embark1],axis=1)


# In[ ]:


test.head(2)


# In[ ]:


train.drop(['Name','Sex','Embarked','Ticket'],axis=1,inplace=True)


# In[ ]:


train.head(2)


# In[ ]:


test.drop(['Name','Sex','Embarked','Ticket'],axis=1,inplace=True)


# In[ ]:


test.head(2)


# In[ ]:


train.drop('PassengerId',axis=1,inplace=True)


# In[ ]:


train.head(2)


# In[ ]:


test.drop('PassengerId',axis=1,inplace=True)


# In[ ]:


test.head(2)


# In[ ]:


X = train.drop('Survived',axis=1)
y = train['Survived']


# In[ ]:


total = train.isnull().sum().sort_values(ascending=False)
percent = train.isnull().sum()/train.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total , percent],axis=1,keys=['Total' , 'Percent'])
missing_data


# In[ ]:


total = X.isnull().sum().sort_values(ascending=False)
percent = X.isnull().sum()/X.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total ,percent],axis=1,keys=['Total','Percent'])
missing_data


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)


# In[ ]:


pred = lr.predict(X_test)
pred


# In[ ]:


testing = pd.read_csv('../input/test.csv')
testing.head(2)


# In[ ]:


my_submission = pd.DataFrame({'survived':lr.predict(test),'passengerId': testing['PassengerId']},index = test.index)


# In[ ]:


my_submission 


# In[ ]:


my_submission.to_csv('submission.csv',index=False)

