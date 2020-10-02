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

import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection,decomposition
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import pickle




# import data 
data = pandas.read_csv('../input/train.csv')
data.head(5)

df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]



# # spliting data
X_train, X_val, y_train, y_val = train_test_split(df_x, df_y, train_size = 0.8,random_state = 42)

# dimension reductionality
pca = decomposition.PCA(n_components=150)
pca.fit(X_train)
PCtrain = pca.transform(X_train)
PCval = pca.transform(X_val)

X_train= PCtrain
X_cv = PCval

# # model train
clf = SVC(kernel='poly',gamma=0.0001, C=100)
clf.fit(X_train,y_train)


# ## For Saving Models
with open('svm.pickle', 'wb') as handle:
    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

  
predicted = clf.predict(X_cv)
expected = y_val


print("Classification report for classifier %s:\n%s\n"
      % (clf, classification_report(expected, predicted)))
print(accuracy_score(expected, predicted))


# In[ ]:




