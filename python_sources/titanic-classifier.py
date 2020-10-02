#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from sklearn.svm import SVC
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df  = pd.read_csv("/kaggle/input/titanic/train.csv")


# In[ ]:


df.head()


# In[ ]:


#### find the missing values
df.isnull().sum()


# In[ ]:


df.shape


# In[ ]:


df  = df.drop('Cabin',axis = 1)


# In[ ]:


df.columns


# In[ ]:


df.isnull().sum()


# In[ ]:


df = df.dropna()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.head()


# In[ ]:


df.drop(['Name','Ticket'] , axis =1, inplace  = True)


# In[ ]:


df.dtypes


# In[ ]:


from sklearn import preprocessing

le_gender = preprocessing.LabelEncoder()
df['Sex'] = le_gender.fit_transform(df['Sex'])


# In[ ]:


le_embarked = preprocessing.LabelEncoder()
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])


# In[ ]:


df.isnull().sum()


# In[ ]:


df.corr()


# In[ ]:


X = df.drop(['Survived','PassengerId'],axis = 1 )
y = df['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#from sklearn.neighbors import KNeighborsClassifier

#model = KNeighborsClassifier(n_neighbors = 5)
#model.fit(X_train, y_train)

model = SVC()
model.fit(X_train, y_train)


# In[ ]:


pred = model.predict(X_test)


# In[ ]:


pred


# In[ ]:


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# In[ ]:


confusion_matrix(y_test,pred)


# In[ ]:


f1_score(y_test,pred)


# In[ ]:


accuracy_score(y_test,pred)


# In[ ]:


test_data  = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


test_data.columns


# In[ ]:


test_data.head()
pass_id = test_data['PassengerId']
print(type(pass_id))
test_data = test_data.drop(['PassengerId','Ticket'],axis =1)


# In[ ]:


### transform
test_data['Sex'] = le_gender.transform(test_data['Sex'])
test_data['Embarked'] = le_embarked.transform(test_data['Embarked'])


# In[ ]:


test_data.drop(['Cabin','Name'],axis =1,inplace = True)


# In[ ]:


test_data.isnull().sum()


# In[ ]:


mean_age =  test_data['Age'].mean()
print(mean_age)

test_data['Age'] = test_data['Age'].fillna(mean_age)


# In[ ]:


mean_fare =  test_data['Fare'].mean()
print(mean_fare)

test_data['Fare'] = test_data['Fare'].fillna(mean_fare)


# In[ ]:


test_data.isnull().sum()


# In[ ]:


submission_df =  pd.read_csv("/kaggle/input/titanic/gender_submission.csv")


# In[ ]:


X_train.columns


# In[ ]:


test_data
pred =  model.predict(test_data)


# In[ ]:


pred


# In[ ]:


len(pass_id)
print(type(pass_id))


# In[ ]:


len(pred)
print(type(pred))


# In[ ]:


pred = pd.DataFrame(pred)
pred.columns = ['Survived']
pred


# In[ ]:


my_submission_df =  pd.concat([pass_id,pred],axis =1)


# In[ ]:


my_submission_df


# In[ ]:


my_submission_df.to_csv("submission.csv")
my_submission_df.columns


# In[ ]:


# this is a sample provided by kaggle present in submission.csv
submission_df

#### 1. remove pass id and pass to model , 2.  output concat with test data and then keep only needed columns

