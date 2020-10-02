#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# * Just import the necessary libraries

# In[32]:


from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# * np.random.seed(20) makes the random numbers predictable
# * With the seed reset (every time), the same set of numbers will appear every time.
# * If the random seed is not reset, different numbers appear with every invocation.

# In[33]:


np.random.seed(20)


# 

# * There are two options first we applied K means clustering and then add the column to the dataset containing(0,1,2) possible classes
# * And the second option is to just replace the every column in the dataset from the K means predicted labels(0,1,2) possible classes
# * Then i have divide the dataset into 70 30 ratio, test data containing 30% of the dataset

# In[34]:


class clust():
    def _load_data(self, sklearn_load_ds):
        data = sklearn_load_ds
        X = pd.DataFrame(data.data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, data.target, test_size=0.3,
                                                                                random_state=42)

    def __init__(self, sklearn_load_ds):
        self._load_data(sklearn_load_ds)

    def classify(self, model=LogisticRegression(random_state=42)):#if no classifier is given then Logistic as default
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print('Accuracy: {}'.format(accuracy_score(self.y_test, y_pred)))

    def Kmeans(self, output='add'):
        n_clusters = len(np.unique(self.y_train))
        clf = KMeans(n_clusters=n_clusters, random_state=42)
        clf.fit(self.X_train)
        y_labels_train = clf.labels_
        y_labels_test = clf.predict(self.X_test)
        if output == 'add':
            self.X_train['km_clust'] = y_labels_train
            self.X_test['km_clust'] = y_labels_test
            print('Train :',self.X_train)
            print('Test :',self.X_test)
            
        elif output == 'replace':
            self.X_train = y_labels_train[:, np.newaxis]
            self.X_test = y_labels_test[:, np.newaxis]
            print('Train :',self.X_train)
            print('Test :',self.X_test)
        else:
            raise ValueError('output should be either add or replace')
        return self


# In[35]:


data=load_iris()

clust(data).Kmeans(output='add').classify(model=RandomForestClassifier())
# clust(data).Kmeans(output='add').classify(model=AdaBoostClassifier())
# clust(data).Kmeans(output='add').classify()


# Achieved 100% accuracy.
