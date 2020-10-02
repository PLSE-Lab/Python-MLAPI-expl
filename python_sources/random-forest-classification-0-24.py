#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')
sample   = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


df_train = df_train.drop(['id'],axis = 1)
df_test  = df_test.drop(['id'],axis = 1)


# In[ ]:


df_train.columns


# In[ ]:


X = df_train.drop('target',axis = 1)
Y = df_train.target


# In[ ]:


rf = RandomForestClassifier(n_estimators = 800,min_samples_leaf = 10,min_samples_split=15)


# In[ ]:


rf.fit(X,Y)


# In[ ]:


y = rf.predict_proba(df_test)


# In[ ]:


y = y[:,1]


# In[ ]:


def calcginiindex(array):

    array = array.flatten()
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1,array.shape[0]+1)
    n = array.shape[0]
    return ((np.sum((2*index - n - 1)*array))/(n * np.sum(array)))


# In[ ]:


val = calcginiindex(y)

print (val)


# In[ ]:


sample['target'] = y
sample.to_csv('submission.csv',index=False)


# In[ ]:




