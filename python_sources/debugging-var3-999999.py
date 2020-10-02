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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

X = training.iloc[:,:-1]
y = training.TARGET

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

selectK = SelectKBest(f_classif, k=220)
selectK.fit(X, y)
X_sel = selectK.transform(X)

features = X.columns[selectK.get_support()]
print (features)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_sel, y, random_state=1301)


# In[ ]:


features[0]


# In[ ]:


training['var3'].value_counts()


# In[ ]:


training.loc[training['var3'] != -999999, 'var3'].hist(bins=1000)


# In[ ]:


X_train = training.loc[training['var3'] != -999999, features[1:]]


# In[ ]:


y_train = training.loc[training['var3'] != -999999, 'var3']


# In[ ]:


X_test = training.loc[training['var3'] == -999999, features[1:]]


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


clf = KNeighborsClassifier(n_neighbors=20)
clf.fit(X_train, y_train)


# In[ ]:


y_test = clf.predict(X_test)


# In[ ]:


y_test


# Conclusion: replace -999999 values with most common values 2 is probably a good approach

# In[ ]:





# In[ ]:




