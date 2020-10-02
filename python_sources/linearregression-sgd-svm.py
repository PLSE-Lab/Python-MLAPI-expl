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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR


# In[ ]:


df = pd.read_csv('../input/Admission_Predict.csv', delimiter=',')


# In[ ]:


X = df.iloc[:,:8].values
y = df.iloc[:, 8].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y)
lr = LinearRegression()


# In[ ]:


lr.fit(X_train, y_train)


# In[ ]:


lr.predict(X_test)


# In[ ]:


lr.score(X_test, y_test)


# In[ ]:


lr.coef_


# In[ ]:


lr.intercept_


# #### SGD

# In[ ]:


standardScaler = StandardScaler()
standardScaler.fit(X_train)


# In[ ]:


X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)


# In[ ]:


sgd_reg = SGDRegressor(n_iter = 50)
sgd_reg.fit(X_train_standard, y_train)


# In[ ]:


sgd_reg.predict(X_test_standard)


# In[ ]:


sgd_reg.coef_


# In[ ]:


sgd_reg.intercept_


# In[ ]:


sgd_reg.score(X_test_standard, y_test)


# ### svm

# In[ ]:


svr = LinearSVR()


# In[ ]:


svr.fit(X_train_standard, y_train)


# In[ ]:


svr.predict(X_test_standard)


# In[ ]:


svr.score(X_test_standard, y_test)


# In[ ]:


svr.coef_


# In[ ]:


svr.intercept_

