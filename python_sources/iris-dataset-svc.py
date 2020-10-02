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
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

iris=datasets.load_iris()

get_ipython().run_line_magic('psearch', 'datasets.load_*')

digits=datasets.load_digits()

print(digits.DESCR)

print(iris.DESCR)

iris.data

iris.data.shape

iris.feature_names

iris.target.shape

iris.target_names

iris_data=pd.DataFrame(iris.data,columns=iris.feature_names)

iris_data.describe()

iris_data['sepal length (cm)'].hist(bins=30)
plt.show()

for class_number in np.unique(iris.target):
  plt.figure(2)
  iris_data['sepal length (cm)'].iloc[np.where(iris.target==class_number)[0]].hist(bins=30)

plt.figure()
plt.subplot(221)
plt.scatter(iris.data[:,0],iris.data[:,1],c=iris.target)
plt.subplot(222)
plt.scatter(iris.data[:,0],iris.data[:,2],c=iris.target)
plt.subplot(223)
plt.scatter(iris.data[:,1],iris.data[:,2],c=iris.target)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test= train_test_split(iris.data[:,:2],iris.target,test_size=0.25,stratify=iris.target,random_state=2)

print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

from sklearn.svm import SVC

clf=SVC(kernel='linear',random_state=1222)

clf.fit(X_train,Y_train)

Y_pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score

acc=accuracy_score(Y_test,Y_pred)

print(acc)

