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


df = pd.read_csv('/kaggle/input/learn-together/train.csv')


# In[ ]:


df = df.set_index(df.Id).drop('Id', axis=1)
y_train = df.Cover_Type
x_train = df.drop('Cover_Type', axis=1)


# In[ ]:


df_test = pd.read_csv('/kaggle/input/learn-together/test.csv')


# In[ ]:


x_test = df_test.set_index(df_test.Id).drop('Id', axis=1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# clf = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=1, max_features=10) # this yielded 0.75779
# clf = RandomForestClassifier(n_estimators=300, criterion='gini', random_state=1, max_features=10, max_leaf_node=15) # this yielded 0.48
clf = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=1, max_features=7) # this yielded 0.75779
clf.fit(x_train, y_train)


# In[ ]:


y_pred = clf.predict(x_test)


# In[ ]:


submission = pd.DataFrame({'Id': df_test['Id'],'Cover_Type': y_pred})


# In[ ]:


filename = 'Forest_Type_Clf0101.csv'

submission.to_csv(filename,index=False)

