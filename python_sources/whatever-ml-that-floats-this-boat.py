#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/titanic/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Get the data into a dataframe to manipulate it better
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()


# In[ ]:


sns.heatmap(train.isnull() , yticklabels=False, cbar=False, cmap='viridis')


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


sns.countplot(x='Survived', data=train, hue='Sex', palette='RdBu_r')


# In[ ]:


sns.countplot(x='Survived', data=train, hue='Pclass')


# In[ ]:


sns.distplot(train['Age'].dropna(), kde=False, bins=30)


# In[ ]:


sns.countplot(x='SibSp', data=train)


# In[ ]:


train['Fare'].hist(bins=40, figsize=(10,4))


# In[ ]:


import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind='hist',bins=50)


# In[ ]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Age',data=train)


# In[ ]:


def impute_age(cols):
    age = cols[0]
    pclass = cols[1]
    if pd.isnull(age):
        if pclass == 1:
            return 37
        if pclass == 2:
            return 29
        if pclass == 3:
            return 24
    else:
        return age
    


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
test.drop('Cabin',axis=1,inplace=True)
test.Fare.fillna(35.627188, inplace=True)


# In[ ]:


sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, embark], axis=1)
test_sex = pd.get_dummies(test['Sex'], drop_first=True)
test_embark = pd.get_dummies(test['Embarked'], drop_first=True)
test = pd.concat([test, test_sex, test_embark], axis=1)


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'], inplace=True, axis=1)
test.drop(['Sex','Embarked','Name','Ticket'], inplace=True, axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.drop(['PassengerId'],inplace=True, axis=1)
test.drop(['PassengerId'],inplace=True, axis=1)


# In[ ]:


y = train['Survived']
X = train.drop(['Survived'], axis=1)


# In[ ]:


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(X, y)


# In[ ]:


predictions = logmodel.predict(test)


# In[ ]:


new_index = np.arange(892,1310)
predictions = pd.DataFrame(data=np.array(predictions), columns=['Survived']) 
predictions.insert(0, 'PassengerID', pd.Series(np.arange(892,1310), index=predictions.index)) 
predictions


# In[ ]:


#from sklearn.metrics import classification_report


# In[ ]:


#print(classification_report(y_test,predictions))


# In[ ]:


#from sklearn.metrics import confusion_matrix


# In[ ]:


#confusion_matrix(y_test, predictions)


# In[ ]:


# Formatting the output
predictions.to_csv('cc_titanic.csv',index=False)
print('saving predictions to csv file <cc_titanic.csv>')

