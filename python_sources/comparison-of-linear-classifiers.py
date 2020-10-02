#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


from sklearn.linear_model import LogisticRegression
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import graphviz


# In[ ]:


class_df = pd.read_csv('/kaggle/input/zoo-animal-classification/class.csv')


# In[ ]:


class_df.head()


# In[ ]:


class_types = class_df['Class_Type']
classes = {}
i=1
for c_type in class_types:
    classes[i] = c_type
    i+=1
    


# In[ ]:


df = pd.read_csv('/kaggle/input/zoo-animal-classification/zoo.csv')


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# In[ ]:


X = df.drop(['animal_name','class_type'], axis=1)
y = df['class_type']


# In[ ]:


column_names = X.columns
feature_names = {}
i = 0

for name in column_names:
    feature_names[i] = name
    i+=1


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


clf = LogisticRegression(random_state=42).fit(X_train, y_train)


# In[ ]:


clf.score(X_test, y_test)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


df['class_type'].value_counts()


# In[ ]:


clf = tree.DecisionTreeClassifier()


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


clf.score(X_test, y_test)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


tree.plot_tree(clf)


# In[ ]:


graph_tree = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(graph_tree)
graph.render('Zoo Animals')


# In[ ]:


dot_data = tree.export_graphviz(clf, out_file=None, 
                                class_names=class_types,
                                feature_names=feature_names,
                                filled=True, rounded=True,  
                                special_characters=True) 
graph = graphviz.Source(dot_data)
graph


# In[ ]:


clf = KNeighborsClassifier()


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


clf.score(X_test, y_test)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


X_train_knn, X_val, y_train_knn, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
for i in range(1,7):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train_knn, y_train_knn)
    print('Validation score for {} neighbours = {}'.format(str(i), str(clf.score(X_val, y_val))))

