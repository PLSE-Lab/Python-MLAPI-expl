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


# ## 1. Data Set Information:
# 
# This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.
# 
# ![](https://archive.ics.uci.edu/ml/assets/MLimages/Large53.jpg)
# 

# In[ ]:


# Check the versions of libraries

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# ## 2. Import Libraries

# In[ ]:


# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ## Load Dataset
# 
# Loading .csv dataset from url provided by kaggle, using Pandas

# In[ ]:


# Load Kaggle filename
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Load dataset
url = "/kaggle/input/iris/Iris.csv"
dataset = read_csv(url)


# ## 3. Sumarize the Dataset

# In[ ]:


# shape
# it has 150 instances and 5 attributes
print(dataset.shape)


# In[ ]:


# head
# print the first 25 rows of the data
print(dataset.head(25))


# In[ ]:


# tail
# print the last 25 rows of the data
print(dataset.tail(25))


# In[ ]:


# descriptions
# use iloc for data selection in Python Pandas. In this case, select all rows and all columns except 'Id'
# print descriptions
print(dataset.iloc[:,1:].describe())


# ## 4. Data cleansing

# In[ ]:


# First, create a dataset backup
dataset_bak = dataset


# In[ ]:


# Remove first column - Id
dataset = dataset.drop('Id',axis=1)
# Head
print(dataset.head(20))


# In[ ]:


# Change column names
dataset.columns = ['Sepal-length', 'Sepal-width', 'Petal-length', 'Petal-width', 'Species']
print(dataset.head(20))


# In[ ]:


# Class distribution, to see the number of rows that belong to each species
print(dataset.groupby('Species').size())


# ## 5. Data Visualization

# ### 5.1. Univariate Plots

# In[ ]:


# Box and whisker plots. Univariate plots, one for each individual variable
fig=plt.figure(figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[ ]:


# Histograms. Create a histogram of each input variable to get an idea of the distribution
dataset.hist()
plt.show()


# ### 5.2. Multivariate Plots

# In[ ]:


# Scatter plot matrix. See all pairs of attributtes, to detect correlations or relationships
scatter_matrix(dataset)
plt.show()


# ## 6. Evaluate Some Algorithms

# - We need to know that the model we created is good.
# - Use statistical methods to estimate the accuray of the models on unseen data: split the dataset in two, 80% to train, evaluate and select among our models and 20% hold back as a validation dataset

# ### 6.1. Create a validation dataset

# In[ ]:


# Split-out validation dataset
array = dataset.values
# All rows and colums except species column
X = array[:,0:4]
# Species column
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# ### 6.2. Build models and evaluate them

# We will use stratified 10-fold cross validation to estimate model accuracy: split the dataset in 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits
# 
# Test 6 different algorithms:
# 
# 1. Logistic Regression (LR)
# 2. Linear Discriminant Analysis (LDA)
# 3. K-Nearest Neighbors (KNN)
# 4. Classification and Regression Trees (CART)
# 5. Gaussian Naive Bayes (NB)
# 6. Support Vector Machines (SVM)

# In[ ]:


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# - Support Vector Machines (SVM) has the largest estimated accuracy score: ~ 98%
# - Linear Discriminant Analysis (LDA) has the second one: ~ 97%

# In[ ]:


# Compare algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()


# ## 7. Making predictions and evaluate them

# In[ ]:


# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)


# In[ ]:


# Evaluate predictions by comparing them to the expected results in the validation set
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

