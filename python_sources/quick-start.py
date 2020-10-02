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


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


wine = pd.read_csv('../input/winequality-red-train.csv')
wine_val = pd.read_csv('../input/winequality-red-validate-data.csv')


# In[ ]:


wine.head()


# In[ ]:


wine.drop(['id'], axis=1, inplace=True)
val = wine_val.drop(['id'], axis=1)
X = wine.drop('quality', axis = 1)
y = wine['quality']


# In[ ]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X, y)


# In[ ]:


pred = rfc.predict(val)
wine_val['quality'] = pred
wine_val.head()


# In[ ]:


wine_val.to_csv('submission.csv', columns=('id', 'quality'), index=False)

