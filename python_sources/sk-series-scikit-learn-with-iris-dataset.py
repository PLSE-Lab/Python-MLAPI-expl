#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import basic libraries
from sklearn import datasets

# load iris dataset
iris = datasets.load_iris()


# In[ ]:


# check iris training data
iris.data


# In[ ]:


# check iris target data(labels)
iris.target


# In[ ]:


# train a SVM model
from sklearn import svm

clf = svm.SVC(gamma=0.1, C=100.)
clf.fit(iris.data, iris.target)


# In[ ]:


# test the model
# just an experiment, we should not use the training data as test data since there is overfitting issue
clf.predict(iris.data)

