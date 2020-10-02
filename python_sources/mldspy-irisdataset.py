#!/usr/bin/env python
# coding: utf-8

# Introduction

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Next Step

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
import warnings

warnings.filterwarnings("ignore")

import os
print(os.listdir("../input//iris"))


# In[ ]:


iris = pd.read_csv("../input/iris/Iris.csv")
iris.head()


# In[ ]:


iris.describe()


# In[ ]:


iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")


# In[ ]:


sns.FacetGrid(iris, hue="Species", size=5)    .map(plt.scatter,"SepalLengthCm", "SepalWidthCm")    .add_legend()


# In[ ]:


sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)


# In[ ]:


sns.boxplot(x="Species", y="PetalLengthCm", data=iris)


# In[ ]:


ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")


# In[ ]:


sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)


# In[ ]:


sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)


# In[ ]:


iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12,6))


# In[ ]:


from pandas.plotting import parallel_coordinates
parallel_coordinates(iris.drop("Id", axis=1), "Species")


# In[ ]:


from pandas.plotting import andrews_curves
andrews_curves(iris.drop("Id", axis=1), "Species")


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# In[ ]:


X = iris.iloc[:,:-1].values
y = iris.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


#LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import accuracy_score
print('accuarcy is',accuracy_score(y_pred,y_test))


# In[ ]:


#K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[ ]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[ ]:


#Support Vector Machine's
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))

