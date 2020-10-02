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
# coding: utf-8

# In[75]:

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import numpy as np


# In[2]:

iris = datasets.load_iris()


# In[38]:

iris_data = iris.data
iris_labels = iris.target


# In[76]:

knn_classifier = KNeighborsClassifier(n_neighbors= 10)
LR_Classifier = LogisticRegression()


# In[70]:

# using cross validation on the data and getting the mean accuaracy 


# In[71]:

cross_validation_data_sets = KFold(len(iris_data), n_folds=10, shuffle=True)


# In[77]:

mean_results_knnclassifier = []
mean_results_LRclassifier = []
for train,test in cross_validation_data_sets:
    train_data = iris_data[train]
    train_label = iris_labels[train]
    test_data = iris_data[test]
    test_label = iris_labels[test]
    knn_classifier.fit(train_data, train_label)
    LR_Classifier.fit(train_data, train_label)
    mean_results_knnclassifier.append(knn_classifier.score(test_data,test_label))
    mean_results_LRclassifier.append(LR_Classifier.score(test_data,test_label))


# In[78]:

mean_results_knnclassifier = np.mean(mean_results_knnclassifier)
mean_results_LRclassifier = np.mean(mean_results_LRclassifier)


# In[80]:

print('mean_results_knnclassifier: ', mean_results_knnclassifier)
print('mean_results_LRclassifier: ', mean_results_LRclassifier)
