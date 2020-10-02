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


df_train=pd.read_csv('/kaggle/input/titanic/train.csv')
df_train.head(1)


# In[ ]:


df_test=pd.read_csv('/kaggle/input/titanic/test.csv')
print (df_test.columns)
df_test.head(2)


# In[ ]:


women = df_train.loc[df_train.Sex == 'female']["Survived"]
#print (women)
rate_women = sum(women)/len(women)
print ('% of women who survived is:',rate_women)
male =df_train.loc[df_train.Sex=='male']['Survived']
rate_male=sum(male)/len(male)
print ('and the % of men who survived is: ', rate_male)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

y = df_train["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(df_train[features])
X_test = pd.get_dummies(df_test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)


output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

