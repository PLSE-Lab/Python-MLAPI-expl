#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # LOGISTIC REGRESSION Vs DECISION TREE CLASSIFIER

# In[ ]:


data=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.head()


# In[ ]:


model=LogisticRegression()


# In[ ]:


from sklearn.model_selection import train_test_split
label=data['target']
data=data.drop('target',axis=1)
train_data,test_data,train_label,test_label=train_test_split(data,label,test_size=0.3)


# In[ ]:


model.fit(train_data,train_label)


# In[ ]:


pred=model.predict(test_data)


# In[ ]:


from sklearn.metrics import accuracy_score
print('Accuracy : ',accuracy_score(test_label,pred))


# In[ ]:


model2=DecisionTreeClassifier()
model2.fit(train_data,train_label)


# In[ ]:


pred2=model2.predict(test_data)


# In[ ]:


print('Accuracy : ',accuracy_score(test_label,pred2))


# In[ ]:




