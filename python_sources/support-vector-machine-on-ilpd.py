#!/usr/bin/env python
# coding: utf-8

# In[55]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[56]:


df = pd.read_csv('../input/Indian Liver Patient Dataset (ILPD).csv')


# In[57]:


df.head(3)


# In[58]:


df.isnull().sum()


# In[59]:


df['A/G'].fillna(df['A/G'].mean(),inplace=True)


# In[60]:


df['A/G'].isnull().sum()


# In[61]:


df.Sex.replace(['Male','Female'],[1,0],inplace=True)


# In[62]:


df.head()


# In[63]:


df.columns


# In[64]:


sns.countplot(df.Target)


# In[65]:


from sklearn.model_selection import train_test_split
X = df[['Age', 'Sex', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G']]
y = df.Target


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42,stratify=df.Target)


# In[67]:


from sklearn.svm import SVC
clf=SVC()
clf.fit(X_train,y_train)


# In[68]:


clf.score(X_test,y_test)


# In[70]:


from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,clf.predict(X_test))


# In[72]:


print(classification_report(y_test,clf.predict(X_test)))

