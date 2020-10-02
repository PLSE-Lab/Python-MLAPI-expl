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


pip install sklearn


# In[ ]:


#import

import numpy as np #linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # statistical data visualization
import matplotlib.pyplot as plt # data visualization

#for model creation
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# import category encoders
import category_encoders as ce


#for decision tree classifcation
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

#for randomforest classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#for knn classification
from sklearn.neighbors import KNeighborsClassifier

#for svm classification
from sklearn.svm import SVC
from sklearn import svm

#for mlp classification
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Load dataset
df = pd.read_csv("../input/car-evaluation/car_evaluation.csv.csv", names = ["buying","maint", "doors", "persons", "lug_boot","safety","class"])
df.head()


# In[ ]:


# view dimensions of dataset
df.shape


# In[ ]:


#view summary of dataset

df.info()


# In[ ]:


#Checking balance factor by exploring the class variable(target variable)
print(df["class"].value_counts())


# In[ ]:


#check missing values in variables

df.isnull().sum()


# In[ ]:


X = df.drop(['class'], axis=1)

y = df['class']


# In[ ]:


#Splitting dataset into training and testing class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[ ]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape


# In[ ]:


#Feature Engineering
#check data types in X_train

X_train.dtypes


# In[ ]:


X_train.head()


# In[ ]:


# encode variables with ordinal encoding

import category_encoders as ce

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

X_train.head()


# In[ ]:


X_train.head()


# In[ ]:


#decision tree classifier with criterion gini index

# instantiate the DecisionTreeClassifier model with criterion gini index

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)


# fit the model
clf_gini.fit(X_train, y_train)


# In[ ]:


#predict the test set results with criterion gini index

y_pred_gini = clf_gini.predict(X_test)


# In[ ]:


#check accuracy score with criterion gini index

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))


# In[ ]:


#Compare the train_set and test_set accuracy

y_pred_train_gini = clf_gini.predict(X_train)

y_pred_train_gini


# In[ ]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))


# In[ ]:


#Check fo overfitting and underfitting

# print the scores on training and test set
# If the two values are wuite comparable then there is no overfitting.

print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))


# In[ ]:


#visualize decision tree

plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(clf_gini.fit(X_train, y_train)) 


# In[ ]:


# Create Decision Tree classifer object with criterion entropy

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=0)

#fit the model
clf = clf.fit(X_train,y_train)

#predict the test set results with criterion entropy
Y_pred = clf.predict(X_test)

#Check accuracy score with criterion entropy
cm = confusion_matrix(Y_pred, y_test)

print("Accuracy of Decision tree classification:",metrics.accuracy_score(y_test, Y_pred))
print(cm)


# In[ ]:


#compare the train set and test set accuracy

y_pred_train_en = clf.predict(X_train)

y_pred_train_en


# In[ ]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))


# In[ ]:


#check for overfitting and underfitting

print('Training set score: {:.4f}'.format(clf.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf.score(X_test, y_test)))


# In[ ]:


#visualize decision trees

plt.figure(figsize=(12,8))

tree.plot_tree(clf.fit(X_train, y_train)) 


# In[ ]:


# Print the Confusion Matrix and slice it into four pieces

cm = confusion_matrix(y_test, Y_pred)

print('Confusion matrix\n\n', cm)


# In[ ]:


#random forest classification
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf = clf.fit(X_train,y_train)

Y_pred = clf.predict(X_test)
cm = confusion_matrix(Y_pred, y_test)

print("Accuracy random froest classification:",metrics.accuracy_score(y_test, Y_pred))
print(cm)


# In[ ]:


#knn clasification
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train,y_train)

Y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_pred, y_test)

print("Accuracy of knn classification:",metrics.accuracy_score(y_test, Y_pred))
print(cm)


# In[ ]:


#support vector machine Classifica
clf = svm.SVC(kernel='linear') 

clf.fit(X_train, y_train)


Y_pred = clf.predict(X_test)
cm = confusion_matrix(Y_pred, y_test)


print("Accuracy of SVM Classifier : ",metrics.accuracy_score(y_test, Y_pred))
print(cm)


# In[ ]:


#MultilayerPercetron classification

clf = MLPClassifier(hidden_layer_sizes=(3,3), max_iter=3000,activation = 'relu',solver='adam',random_state=1)
clf=clf.fit(X_train, y_train)

cm = confusion_matrix(Y_pred, y_test)
Y_pred = clf.predict(X_test)

print("Accuracy of MLPClassifier : ",metrics.accuracy_score(y_test, Y_pred))
print(cm)

