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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


train.tail()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[ ]:


train['Age'].hist(bins=30,color='darkred',alpha=0.7)


# In[ ]:


sns.countplot(x='SibSp',data=train)


# In[ ]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[ ]:



sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass ==1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


train.info()


# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test =train_test_split(train.drop('Survived',axis=1),
                                               train['Survived'],test_size=0.2,random_state=101)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)


# In[ ]:


rf_pre = rf.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,rf_pre))
print(classification_report(y_test,rf_pre))


# In[ ]:


test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


sns.heatmap(test.isnull())


# In[ ]:


test.drop('Cabin',axis=1,inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)


# In[ ]:


test.info()


# In[ ]:


test.head()


# In[ ]:


test['Age']=test[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sex_test = pd.get_dummies(test['Sex'],drop_first=True)
embark_test=pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex_test,embark_test],axis=1)
test.head()


# In[ ]:


test_prediction = rf.predict(test)
test_prediction


# In[ ]:


test_pred = pd.DataFrame(test_prediction,columns=['Survived'])
new_test = pd.concat([test,test_pred],axis=1,join='inner')
new_test.head()


# In[ ]:


df = new_test[['PassengerId','Survived']]
df.to_csv('predictions.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:




