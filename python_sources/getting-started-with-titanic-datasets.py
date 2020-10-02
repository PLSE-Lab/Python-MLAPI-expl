#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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


train_X = pd.read_csv("../input/titanic/train.csv")
test_y = pd.read_csv("../input/titanic/test.csv")
print("Test and Train FILES are loaded")
train_X.columns


# In[ ]:


from sklearn.preprocessing import LabelEncoder
features = ["Pclass", "Sex", "SibSp", "Parch"]
train_y = train_X['Survived']
train_f= train_X[features]
le = LabelEncoder()
train_f['Sex']=le.fit_transform(train_f['Sex'])
test_y['Sex'] = le.fit_transform(test_y['Sex'])


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(train_f,train_y)
predictions = model.predict(test_y[features])
predictions


# In[ ]:



output = pd.DataFrame({"PassengerId" :test_y['PassengerId'] , 'Survived':predictions})
output.to_csv('submission1.csv',index=False)
print("Your submission was successfully saved!")

