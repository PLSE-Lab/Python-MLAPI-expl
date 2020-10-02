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


# Author: Bernhard Pfahringer , ID: 1234567

# In[ ]:


iris = pd.read_csv('../input/Iris.csv')
iris.head()


# Get some more info about the data:

# In[ ]:


iris.info()


# Plotting distributions is always a good idea

# In[ ]:


import seaborn as sns
sns.pairplot(data=iris, hue='Species', palette='Set2')


# Split the data into X and Y, input and output, and then into training and test set

# In[ ]:


from sklearn.model_selection import train_test_split
x = iris.iloc[:, 1:-1]
y = iris.iloc[:, 5]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(x_train.shape, y_train.shape)


# Import a standard learning algorithm, instantiate it, and train it on the training data

# In[ ]:


from sklearn.svm import SVC
model=SVC()
model.fit(x_train, y_train)


# Use the model to predict for the test data, and check a few prediction manually

# In[ ]:


pred = model.predict(x_test)
print(pred[:5])
print(y_test[:5])


# That looked ok, but how good is the model?

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred))


# In[ ]:


print(classification_report(y_test, pred))


# That went well, a perfect result :-)
