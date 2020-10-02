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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import matplotlib.pyplot as pd
import pandas as pd
import os


# In[ ]:


dataset = pd.read_csv('../input/epldata_final.csv')


# In[ ]:


X = dataset.iloc[:, 0:10].values
Y = dataset.iloc[:, 16].values

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
X[:, 0] = lb.fit_transform(X[:, 0]) 
lb1 = LabelEncoder()
X[:, 1] = lb1.fit_transform(X[:, 1])
lb2 = LabelEncoder()
X[:, 3] = lb2.fit_transform(X[:, 3])
lb3 = LabelEncoder()
X[:, 8] = lb3.fit_transform(X[:, 8])


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)


# In[ ]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train, Y_train)


# In[ ]:


Y_pred = lm.predict(X_test)
from sklearn.model_selection import cross_val_score
cvs = cross_val_score(estimator = lm,
                     X = X_train,
                     y = Y_train,
                     cv = 10)
cvs.mean()

