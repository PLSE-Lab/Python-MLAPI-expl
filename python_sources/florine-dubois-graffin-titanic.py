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


import numpy as np
import pandas as pd


# In[ ]:


get_ipython().system('ls ../input/titanic')


# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")


# In[ ]:


gender_submission.head()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


data = pd.concat([train, test], sort=False)


# In[ ]:


data.head()


# In[ ]:


print(len(train), len(test), len(data))


# In[ ]:


data.isnull().sum()


# In[ ]:


data['Sex'].replace(['male','female'], [0, 1], inplace=True)


# In[ ]:


data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[ ]:


data['Fare'].fillna(np.mean(data['Fare']), inplace=True)


# In[ ]:


age_avg = data['Age'].mean()
age_std = data['Age'].std()

data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)


# In[ ]:


delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)


# In[ ]:


train = data[:len(train)]
test = data[len(train):]


# In[ ]:


y_train = train['Survived']
X_train = train.drop('Survived', axis = 1)
X_test = test.drop('Survived', axis = 1)


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


y_pred[:20]


# In[ ]:


sub = gender_submission
sub['Survived'] = list(map(int, y_pred))
sub.to_csv("submission.csv", index=False)


# In[ ]:




