#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


iris = pd.read_csv("../input/Iris.csv")


# In[ ]:


iris.head()


# Below line of code is to drop the Id column, as we gain no information from it. Axis 1 speicifes vertical, and inplace means the changes happen on the dataframe itself instead of creating a new copy.

# In[ ]:


iris.drop('Id',axis=1,inplace=True)


# In[ ]:


fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# In[ ]:


fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# Trying out 3 classifier, Logistic Regression, SVM and Decision Trees.

# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFECV, RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[ ]:


train, test = train_test_split(iris, test_size = 0.4)
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_Y = train.Species
test_Y = test.Species
train_X.head()


# In[ ]:


mutual_info_classif(train_X, train_Y)


# Using mutual information to select which features to include for classification. We can clearly see that petal width and petal length are both not needed as they provide similar information. 

# In[ ]:


train_RFE = train[['SepalWidthCm', 'PetalWidthCm', 'SepalLengthCm']]
test_RFE = test[['SepalWidthCm', 'PetalWidthCm', 'SepalLengthCm']]
train_RFE.head()


# In[ ]:


mutual_info_classif(train_RFE, train_Y)


# Even after removing petal length we see that petal width is also an unnecessary feature as we gain no new information from it. 

# In[ ]:


train_final = train[['SepalWidthCm', 'SepalLengthCm']]
test_final = test[['SepalWidthCm', 'SepalLengthCm']]
train_final.head()


# **Logistic Regression**

# In[ ]:


model = LogisticRegression()
model.fit(train_X,train_Y)
prediction = model.predict(test_X)
print('Accuracy: ', metrics.accuracy_score(prediction, test_Y))
print('Confusion matrix: ', metrics.confusion_matrix(prediction, test_Y))


# In[ ]:


model = LogisticRegression()
model.fit(train_final,train_Y)
prediction = model.predict(test_final)
print('Accuracy: ', metrics.accuracy_score(prediction, test_Y))
print('Confusion matrix: ', metrics.confusion_matrix(prediction, test_Y))


# In[ ]:


model = LogisticRegression()
model.fit(train_RFE,train_Y)
prediction = model.predict(test_RFE)
print('Accuracy: ', metrics.accuracy_score(prediction, test_Y))
print('Confusion matrix: ', metrics.confusion_matrix(prediction, test_Y))


# **SVM**

# In[ ]:


model = svm.SVC()
model.fit(train_X,train_Y)
prediction = model.predict(test_X)
print('Accuracy: ', metrics.accuracy_score(prediction, test_Y))
print('Confusion matrix: ', metrics.confusion_matrix(prediction, test_Y))


# In[ ]:


model = svm.SVC()
model.fit(train_final,train_Y)
prediction = model.predict(test_final)
print('Accuracy: ', metrics.accuracy_score(prediction, test_Y))
print('Confusion matrix: ', metrics.confusion_matrix(prediction, test_Y))


# In[ ]:


model = svm.SVC()
model.fit(train_RFE,train_Y)
prediction = model.predict(test_RFE)
print('Accuracy: ', metrics.accuracy_score(prediction, test_Y))
print('Confusion matrix: ', metrics.confusion_matrix(prediction, test_Y))


# **Decision Tree**

# In[ ]:


model = DecisionTreeClassifier()
model.fit(train_X,train_Y)
prediction = model.predict(test_X)
print('Accuracy: ', metrics.accuracy_score(prediction, test_Y))
print('Confusion matrix: ', metrics.confusion_matrix(prediction, test_Y))


# In[ ]:


model = DecisionTreeClassifier()
model.fit(train_final,train_Y)
prediction = model.predict(test_final)
print('Accuracy: ', metrics.accuracy_score(prediction, test_Y))
print('Confusion matrix: ', metrics.confusion_matrix(prediction, test_Y))


# In[ ]:


model = DecisionTreeClassifier()
model.fit(train_RFE,train_Y)
prediction = model.predict(test_RFE)
print('Accuracy: ', metrics.accuracy_score(prediction, test_Y))
print('Confusion matrix: ', metrics.confusion_matrix(prediction, test_Y))


# We see that removing petal length gives almost the same result as keeping it. For datasets with lots of features, selection helps in reducing the complexity of the algorithm. 
