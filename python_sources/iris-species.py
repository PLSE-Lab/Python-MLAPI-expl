#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import iris_helper as H

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Initialize**
# 
# Let's start with importing the dataset
# 

# In[ ]:


iris = pd.read_csv("../input/iris/Iris.csv")
print(iris.columns)
print(iris.describe())


# Here *Id* is the index and *Species* is the label. Let's look at the first row

# In[ ]:


print(iris.head(1))


# We do not need the *Id* column. So let's drop it and see how our first row looks.

# In[ ]:


iris.drop("Id", axis=1, inplace=True)
iris.head(1)


# We will now take the first 4 columns into our numpy array for *X* and the last column into our numpy array *y*. *X* contains our features and *y* contains our label. Let's also map each of the flower species to a class.
# 
# Iris-setosa -> 0
# 
# Iris-versicolor -> 1
# 
# Iris-virginica -> 2

# In[ ]:


X = iris.iloc[:,:4]
labels = iris.iloc[:,4].unique()
species = dict()
label = 0
for i in labels:
    species[i] = label
    label+=1
y = iris.iloc[:,4].map(species)


# We will split our dataset into 3 parts: Training, Dev and Test. Training set will have 100 samples, and Dev and Test will have 25 samples each.

# In[ ]:


X_train, X_test, y_train, y_test = H.create_test_set(X, y)
X_train, X_dev, y_train, y_dev = H.create_dev_set(X_train, y_train)


# We will now run Standard Scaler on our data.

# In[ ]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_dev = sc.transform(X_dev)
X_test = sc.transform(X_test)


# **Initial Run**
# 
# We will now run initial versions of our models on training data. I chose the following 7 models:
# 1. Decision Tree Classifier
# 2. Gaussian Naive Bayes
# 3. K Nearest Neighbors (KNN)
# 4. Logistic Regression
# 5. MLP Classifier
# 6. Random Forest Classifier
# 7. SVM Classifier
# 
# We will save the cross validation score of each one of them and display in a tabular format for easy analysis.

# In[ ]:


# run initial versions on training data

results=list()
columns = ["CVS_Mean", "CVS_Std (x1e-4)", "Time (ms)"]
index = ["Decision Tree", "Gaussian NB", "KNN", "Logistic Reg", "MLP", "Random Forest", "SVM"]
clf, _ = H.build_decision_tree()
results.append(H.initial_run(clf, X_train, y_train))
clf, _ = H.build_gnb()
results.append(H.initial_run(clf, X_train, y_train))
clf, _ = H.build_knn()
results.append(H.initial_run(clf, X_train, y_train))
clf, _ = H.build_log_reg()
results.append(H.initial_run(clf, X_train, y_train))
clf, _ = H.build_mlp()
results.append(H.initial_run(clf, X_train, y_train))
clf, _ = H.build_random_forest()
results.append(H.initial_run(clf, X_train, y_train))
clf, _ = H.build_svm()
results.append(H.initial_run(clf, X_train, y_train))
initial_run = pd.DataFrame(results, columns=columns, index=index)
print(initial_run)


# **Fine Tuning models on the training set**
# 
# We choose Decision Tree, MLP, SVM and Gaussian Naive Bayes for fine tuning usnig GridSearchCV as they have the highest CVS score.

# In[ ]:


## We run GridSearch on them to fine tune them

best_decision_tree = H.find_best_decision_tree(X_train, y_train)
best_svm = H.find_best_svm(X_train,y_train)
best_mlp = H.find_best_mlp(X_train,y_train)
best_gnb = H.find_best_gnb(X_train,y_train)


# In[ ]:


#
### Now we see the performance of our fine tuned models on training and CV sets
#
results=list()
del results
results=list()
columns = ["CVS_Mean", "CVS_Std (x1e-4)", "Time (ms)"]
index = ["Decision Tree","SVM", "MLP", "GNB"]
results.append(H.best_cvs(best_decision_tree, X_train, y_train))
results.append(H.best_cvs(best_svm, X_train, y_train))
results.append(H.best_cvs(best_mlp, X_train, y_train))
results.append(H.best_cvs(best_gnb, X_train, y_train))
best_cvs = pd.DataFrame(results, columns=columns, index=index)
print(best_cvs)


# **Accuracy on the dev set**
# 
# SVM, MLP and Gaussian NB gave similar results, but Decision Tree Classifier performed quite poorly. So we then test the first three on the dev set, to see if there is any overfitting issue.

# In[ ]:


#
### We find the accuracy of our models on our dev set
#
columns = ["Accuracy", "Time (us)"]
index = ["SVM", "MLP", "GNB"]
del results
results = list()
results.append(H.find_accuracy(best_svm, X_dev, y_dev))
results.append(H.find_accuracy(best_mlp, X_dev, y_dev))
results.append(H.find_accuracy(best_gnb, X_dev, y_dev))
accuracy = pd.DataFrame(results, columns=columns, index=index)
print(accuracy)


# **Accuracy on the testing set**
# 
# All the samples in the dev set were predicted correctly. So we then move on to the test set.

# In[ ]:


#
### We find the accuracy of our models on our test set
#
columns = ["Accuracy", "Time (us)"]
index = ["SVM", "MLP", "GNB"]
del results
results = list()
results.append(H.find_accuracy(best_svm, X_test, y_test))
results.append(H.find_accuracy(best_mlp, X_test, y_test))
results.append(H.find_accuracy(best_gnb, X_test, y_test))
accuracy = pd.DataFrame(results, columns=columns, index=index)
print(accuracy)


# **Conclusion**
# 
# Gaussian NB missed one sample, but SVM and MLP were spot on again. Since SVM is much faster than MLP while training, SVM can be preferred if the model needs to be re-trained frequently. Else, both are equally good.

# In[ ]:





# In[ ]:




