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


import matplotlib.pyplot as plt
plt.style.use("ggplot")


# In[ ]:


df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")


# In[ ]:


df.head()


# In[ ]:


X = df.drop("target", axis=1)
y = df['target']


# In[ ]:


y.hist(bins=2)


# In[ ]:


X.head()


# In[ ]:


X.describe()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=250)


# In[ ]:


rfc.fit(X_train, y_train)


# In[ ]:


preds = rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[ ]:


accuracy_score(y_test, preds)


# In[ ]:


print(classification_report(y_test, preds))


# In[ ]:


confusion_matrix(y_test, preds)


# In[ ]:




