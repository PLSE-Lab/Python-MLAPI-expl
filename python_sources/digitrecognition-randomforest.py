#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data.shape


# In[ ]:


test_data.shape


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().any().sum()


# In[ ]:


train_data['label'].value_counts().sort_values(ascending=True)


# In[ ]:


labeled_data = train_data['label']


# In[ ]:


train_data.drop('label', inplace=True, axis=1)
train_data.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_data, labeled_data, test_size=0.2, random_state=2)


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


X_test.head()


# In[ ]:


y_test.head()


# In[ ]:


rfc  = RandomForestClassifier(n_estimators = 300, n_jobs=-1)


# In[ ]:


rfc.fit(X_train, y_train)


# In[ ]:


rfc_pred = rfc.predict(X_test)


# In[ ]:


confusion_matrix = confusion_matrix(y_test, rfc_pred)
confusion_matrix


# In[ ]:


accuracy_score(y_test, rfc_pred)


# In[ ]:


classification_report = classification_report(y_test, rfc_pred)
print(classification_report)


# In[ ]:


submissions=pd.DataFrame({"ImageId": list(range(1,len(rfc_pred)+1)),
                         "Label": rfc_pred})
submissions.to_csv("submission.csv", index=False, header=True)

