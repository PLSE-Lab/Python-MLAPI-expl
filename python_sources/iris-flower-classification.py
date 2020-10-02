#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classication
# 

# This is one of the most famous dataset in data science. The task is to classify the sample into three types iris flower-Iris-setosa, Iris-versicolor, and Iris-virginica based on sepal length, sepal width, petal length, and petal width. This is a classic multi-variable multi-class classification problem.

# ## Import Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', None)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Import Dataset

# In[ ]:


iris = pd.read_csv('/kaggle/input/iris/Iris.csv')


# In[ ]:


iris.head()


# In[ ]:


iris.info()


# In[ ]:


iris.describe()


# ## Initial Exploration
# Before running any algorithms, let's first try to understand the dataset a bit.

# First, let's check out the variables, and see if there's any correlation with each other.

# In[ ]:


iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].corr()


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].corr(), vmin=-1.0, vmax=1.0, annot=True, linewidths=2)
plt.show()


# We can see that there's a really strong positive correlation between petal length and petal width, a strong positive correlation between sepal length and petal length, and a strong positive correlation between sepal length and petal width. If there's a huge dataset with large amount of features, we can run a dimensionality reduction algorithm to take out the redundant features, but because this is such a small dataset, we can keep all the features.

# Now let's take a look at the distribution of each variables with respect to each type of Iris.

# In[ ]:


iris.groupby('Species').describe()


# In[ ]:


plt.figure(figsize=(8,8))
ax = sns.boxplot(x="Species", y="SepalLengthCm", data=iris).set_title('Sepal Length')
plt.show()


# In[ ]:


plt.figure(figsize=(8,8))
ax = sns.boxplot(x="Species", y="SepalWidthCm", data=iris).set_title('Sepal Width')
plt.show()


# In[ ]:


plt.figure(figsize=(8,8))
ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris).set_title('Petal Length')
plt.show()


# In[ ]:


plt.figure(figsize=(8,8))
ax = sns.boxplot(x="Species", y="PetalWidthCm", data=iris).set_title('Petal Width')
plt.show()


# From the four graphs above, we can see that there's a large difference in distribution of data especially for sepal length, petal length, petal width, not so much for sepal width. Decision tree seems like a sensible algorithm to model the problem.

# ## Modeling
# 

# We start with separating the dataset into training, validation, and test set.

# In[ ]:


# We take 80% of data into training, and 20% into test
# For each set, a third belonds to each type of Iris
iris.drop(['Id'], axis=1, inplace=True)
training = pd.concat([iris[:40], iris[50:90], iris[100:140]])
test = pd.concat([iris[40:50], iris[90:100], iris[140:]])
training_X = training[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
training_y = training['Species']
test_X  = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
test_y = test['Species']


# In[ ]:


print('Training set:', training_X.shape)
print('Test set:', test_X.shape)


# We will run a couple classification algorithms on the dataset and see how well they work.
# * Logistic Regression
# * Decision Tree
# * K Nearest Neighbor(KNN)
# * Support Vector Machine

# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
LR_classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200).fit(training_X, training_y)
print('Training accuracy:', LR_classifier.score(training_X, training_y))
print('Test accuracy:', LR_classifier.score(test_X, test_y))


# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dTree_classifier = DecisionTreeClassifier(criterion="entropy").fit(training_X, training_y)
print('Training accuracy:', dTree_classifier.score(training_X, training_y))
print('Test accuracy:', dTree_classifier.score(test_X, test_y))


# ### K Nearest Neighbor(KNN)

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
KNN_classifier = KNeighborsClassifier().fit(training_X, training_y)
print('Training accuracy:', KNN_classifier.score(training_X, training_y))
print('Test accuracy:', KNN_classifier.score(test_X, test_y))


# ### Support Vector Machine

# In[ ]:


from sklearn.svm import LinearSVC
SVC_classifier = LinearSVC(multi_class='crammer_singer', max_iter=3000).fit(training_X, training_y)
SVC_classifier.score(training_X, training_y)
print('Training accuracy:', SVC_classifier.score(training_X, training_y))
print('Test accuracy:', SVC_classifier.score(test_X, test_y))


# All four algorithms performs perfectly on the test set, achieving 100% accuracy. We can now use the algorithms to predict the type of Iris when we have new measurements.

# At last, let's visualize the decision tree.

# In[ ]:


from sklearn.tree import plot_tree
plt.figure(figsize=(10,10))
plot_tree(dTree_classifier)
plt.show()

