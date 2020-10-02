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
for dirname, _, filenames in os.walk('..'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/learn-together/train.csv', index_col = 0)
print(data.columns)
X = data.drop('Cover_Type',axis=1)
y = data['Cover_Type']

X.describe()


# Now, let's try some different parameters to see what forest configuration gives the best results. As a little review, here's what we're varying in each case:
# 
# **n_estimators** - the number of trees in the forest to use
# 
# **criterion** - how the alg decides when it's apporpriate to make splits in the tree
# 
# **min_samples_leaf **- since they are decimals, they represent the proportion of the sample that may be left on a leaf

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rfc = RandomForestClassifier()
params = {'n_estimators': [20,50,100,150,200,250], 'criterion': ['gini', 'entropy'], 'min_samples_leaf' : [1,.01,.001]}

mysearch = GridSearchCV(rfc, params, cv=5)
mysearch.fit(X,y)


# In[ ]:


mysearch.best_estimator_


# In[ ]:


testdata = pd.read_csv('../input/learn-together/test.csv', index_col = 0)
print(testdata.tail)
results = mysearch.predict(testdata)


# In[ ]:


predictions = pd.DataFrame({'Id': testdata.index, 'Cover_Type':results})
predictions.tail
predictions.to_csv('submission.csv', index= False)


# In[ ]:




