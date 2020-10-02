#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In this notebook, I will do the Digit recognition using 2 algorithms: Support Vector machines and Random forests. First, let's import the required libraries.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import missingno
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Once, we have imported the libraries, let's get the training dataset.

# In[ ]:


df = pd.read_csv('../input/digit-recognizer/train.csv')
df.head()


# There are 785 columns in the dataset, with the 1st column being the output column. So, let's first see how the pixel values look like when plotted. We will use matplotlib for the plotting purposes.

# In[ ]:


label = df.iloc[0, 1:].values
label=label.reshape(28,28)
plt.imshow(label, cmap = 'gray')


# Clearly, it is a very nicely written dataset. Now, let's first start with classification using the Random forest classifier. We will use scikit learn for these purposes. Let's first divide the dataset into inputs and outputs. After that, we will train the classifier.

# In[ ]:


X = df.iloc[:, 1:].values
Y = df.iloc[:, 0].values


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.3, random_state = 5)
clf = RandomForestClassifier(n_estimators=200,max_samples=0.5)
clf.fit(X_train, Y_train)
print(clf.score(X_train, Y_train))


# The score comes to be ~99.9% which is very good for practical purposes. This was when the train_test split was done with only 30% data taken as train_data. If this ratio is increased, it will push the accuracy further up.
# 
# Now, let's start with the SVM classifier. We will use the one-vs-one SVM classifier provided by the scikit learn library

# In[ ]:


clf2 = svm.SVC(decision_function_shape='ovo')


# In[ ]:


clf2.fit(X_train, Y_train)


# In[ ]:


clf2.score(X_train, Y_train)


# Although, it's not better than the Random forest classifier, but will reach the accuracy if we increase the train_data to >30%. Now, use this classifier to predict the test_data. So, first import the test_data from the test.csv file. Then for any random example in the dataset, plot the pixels and see the answers given by the classifier.

# In[ ]:


df_test = pd.read_csv('../input/digit-recognizer/test.csv')
df_test.head()


# In[ ]:


i = 5
plt.imshow(df_test.iloc[i, :].values.reshape(28,28), cmap = 'gray')
print(clf.predict(df_test.iloc[i,:].values.reshape(1,-1)))
print(clf2.predict(df_test.iloc[i,:].values.reshape(1,-1)))


# Clearly, both the classifiers are doing very good on classifying the data.
