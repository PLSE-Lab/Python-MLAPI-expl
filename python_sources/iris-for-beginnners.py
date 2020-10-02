#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/Iris.csv')


# In[ ]:


data.info()


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize = (20, 7))
ax = sns.scatterplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data = data, hue = 'Species', ax = ax[0])
ax1 = sns.scatterplot(x = 'PetalLengthCm', y = 'PetalWidthCm', data = data, hue = 'Species')


# The data is scatter plotted and the three classes are represented here.

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize = (20, 7))
ax = sns.regplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data = data, ax = ax[0])
ax1 = sns.regplot(x = 'PetalLengthCm', y = 'PetalWidthCm', data = data)


# The lines represent the linear regression fit in the data and the adjascent coloured area represent approximate confidenceinterval for the data.
# Here, we can see that there is a more or less linear relationship between the dimensions of the petals, unlike the sepals.

# In[ ]:


d = data.drop('Id', axis = 1)
fig = plt.figure(figsize = (10, 10))
ax = sns.pairplot(d, hue= 'Species')


# In[ ]:


data.drop('Id', axis = 1, inplace = True)
data.info()


# Creating Machine Learning Model

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


train, test = train_test_split(data, test_size = 0.3)
print(train.shape)
print(test.shape)


# In[ ]:


train_X = train.drop('Species', axis = 1)
train_y = train['Species']
test_X = test.drop('Species', axis = 1)
test_y = test['Species']


# In[ ]:


test_X.head()


# In[ ]:


model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print("Accuracy = {}".format(metrics.accuracy_score(prediction, test_y)))


# In[ ]:


model = svm.SVC()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print("Accuracy = {}".format(metrics.accuracy_score(prediction, test_y)))


# In[ ]:


model = DecisionTreeClassifier()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print("Accuracy = {}".format(metrics.accuracy_score(prediction, test_y)))


# In[ ]:


model = KNeighborsClassifier(n_neighbors=2)
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print("Accuracy = {}".format(metrics.accuracy_score(prediction, test_y)))


# In[ ]:




