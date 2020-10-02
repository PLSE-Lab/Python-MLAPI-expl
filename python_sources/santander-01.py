#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


display(train_df.head())
display(train_df.describe())


# In[ ]:


train_null_s = train_df.isnull().sum()
test_null_s = test_df.isnull().sum()
print(train_null_s[train_null_s>0])
print(test_null_s[test_null_s>0])


# In[ ]:


sns.countplot(train_df["target"])


# In[ ]:


train_df.columns.values[2:202]


# In[ ]:


features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']


# In[ ]:


X_train = train_df.drop(['ID_code', 'target'],axis=1)
y_train = train_df["target"]


# In[ ]:


y_pred = rfc.predict(test_df[])


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train,y_train)


# In[ ]:


X_test = test_df.drop('ID_code',axis=1)
y_test_pred = rfc.predict(X_test)


# In[ ]:


print(y_test_pred)


# In[ ]:


df_sub = pd.read_csv("./submission.csv")
sns.countplot(df_sub["target"])


# In[ ]:


my_submission = pd.DataFrame({'ID_code': test_df["ID_code"].values, 'target': y_test_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[ ]:




