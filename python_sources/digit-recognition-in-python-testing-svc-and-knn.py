#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Read sample_submission files
sample_submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
sample_submission.head()


# In[ ]:


# Read train files
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train.head()


# In[ ]:


# Show some images
for x in range(0, 20):
    image = train.loc[x,train.columns != "label"]
    plt.imshow(np.array(image).reshape((28, 28)), cmap="gray")
    plt.show()
    
    plt.hist(image)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Counts")
    plt.show()


# In[ ]:


# Number of train images
print("Number of images: %d" % len(train))
train.head()


# **sklearn.model_selection.train_test_split(*arrays, **options)[source]**
# 
# 
# Split arrays or matrices into random train and test subsets
# 
# Quick utility that wraps input validation and next(ShuffleSplit().split(X, y)) and application to input data into a single call for splitting (and optionally subsampling) data in a oneliner.
# 
# 
# >>> X
# array([[0, 1],
#        [2, 3],
#        [4, 5],
#        [6, 7],
#        [8, 9]])
# 
# 
# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
# 
# >>> X_train
# array([[4, 5],
#        [0, 1],
#        [6, 7]])
#        
#        
# >>> y_train
# [2, 0, 3]
# 
# 
# >>> X_test
# array([[2, 3],
#        [8, 9]])
#        
#        
# >>> y_test
# [1, 4]

# In[ ]:


# split the data
train_images = train.loc[:, train.columns != "label"] / 255
train_labels = train.label


#Split arrays or matrices into random train and test subsets
#Quick utility that wraps input validation and next(ShuffleSplit().split(X, y)) and application to input data into a single call for splitting (and optionally subsampling) data in a oneliner.
x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=1)


# In[ ]:


print ('y_test ****************')
print(y_test)
print ('x_test ****************')
print(x_test)
print ('y_train ****************')
print(y_train)
print ('y_train ****************')
print(y_train)


# **SVC / SVM **
# 
# Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.
# 
# The advantages of support vector machines are:
# 
# Effective in high dimensional spaces.
# Still effective in cases where number of dimensions is greater than the number of samples.
# Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
# Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
# The disadvantages of support vector machines include:
# 
# If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
# SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).
# The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
# 
# 
# SVC, NuSVC and LinearSVC are classes capable of performing multi-class classification on a dataset.

# In[ ]:


# this takes about 20 minutes. accuray = 94.0

#SVC classifier
model = SVC()
model.fit(x_train, y_train)

# this takes about 5 min also
test_predicts = model.predict(x_test)
print(test_predicts)

from sklearn.metrics import accuracy_score
test_acc = round(accuracy_score(y_test, test_predicts) * 100)


# In[ ]:


test_acc


# **Random forests**
# 
# 
# Random forests are an ensemble learning method that can be used for classification. It works by using a multitude of decision trees and it selects the class that is the most often predicted by the trees.
# 
# A decision tree contains at each vertex a "question" and each descending edge is an "answer" to that question. The leaves of the tree are the possible outcomes. A decision tree can be built automatically from a training set.
# 
# Each tree of the forest is created using a random sample of the original training set, and by considering only a subset of the features (typically the square root of the number of features). The number of trees is controlled by cross-validation.

# In[ ]:


# KNN - RandomForestClassifier
# this takes about 10 minutes. accuray = 96.0

modelRandomForestClassifier = RandomForestClassifier(n_estimators=100)
modelRandomForestClassifier.fit(x_train, y_train)
testRandomForestClassifier_predicts = modelRandomForestClassifier.predict(x_test)
test_acc_RandomForestClassifier = round(accuracy_score(y_test, testRandomForestClassifier_predicts) * 100)


# In[ ]:


test_acc_RandomForestClassifier


# * 96% acc for RandomForestClassifier
# * 94% acc for SVC

# In[ ]:


# Read test files & apply model & write the output in a submission file
test_for_submission = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_for_submission.head()

test = test_for_submission.loc[:, :] / 255
submit = modelRandomForestClassifier.predict(test)
pd.DataFrame(submit).to_csv('submit.csv', index=False) 

