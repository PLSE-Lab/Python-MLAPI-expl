#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/Iris.csv')


# In[ ]:


X = data.ix[:, data.columns != 'Species']
y = data.ix[:, data.columns == 'Species']


# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# le.fit_transform(y)

# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.3)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier(n_estimators=10)


# In[ ]:


rf.fit(X_train,y_train)


# In[ ]:


y_test_predicted = rf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test, y_test_predicted)


# In[ ]:




