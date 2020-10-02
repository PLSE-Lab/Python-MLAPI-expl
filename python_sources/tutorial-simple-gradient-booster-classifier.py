#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()


# In[ ]:


Y = df['Survived']
df = df.drop(labels = 'Survived', axis = 1)


# In[ ]:


Y.head()


# In[ ]:


df.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2, shuffle=True)


# In[ ]:


X_train.head()


# In[ ]:


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X_train = pd.get_dummies(X_train[features])
X_test = pd.get_dummies(X_test[features])


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1)


# In[ ]:


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.fit_transform(X_test)


# In[ ]:


GBC.fit(X_train, y_train)
predict = GBC.predict(X_train)


# In[ ]:


from sklearn.metrics import accuracy_score
eval = accuracy_score(y_train, predict)
print(eval)


# In[ ]:


predict = GBC.predict(X_test)
eval = accuracy_score(y_test, predict)
print(eval)


# In[ ]:


X_comp = pd.read_csv('/kaggle/input/titanic/test.csv')
X_comp.head()
X_comp = pd.get_dummies(X_comp[features])


# In[ ]:


X_comp.head()


# In[ ]:


X_comp = my_imputer.fit_transform(X_comp)


# In[ ]:


predict = GBC.predict(X_comp)


# In[ ]:


x_test = pd.read_csv('/kaggle/input/titanic/test.csv')
x_test.head()


# In[ ]:


output = pd.DataFrame({'PassengerId': x_test.PassengerId, 'Survived': predict})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:





# In[ ]:





# In[ ]:


kaggle competitions submit -c titanic -f submission.csv -m "v2.0"

