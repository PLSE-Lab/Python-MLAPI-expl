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


#import the data
diabetes = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")


# In[ ]:


#gets name of column
diabetes.columns


# In[ ]:


#information about the column
diabetes.info()


# In[ ]:


#get basic statistics about the data
diabetes.describe()


# In[ ]:


# check for the null values
diabetes.isnull().sum()


# ## Conclusion 1
# 
# It seems there is no columns with null values

# In[ ]:


#let us look at first 10 rows
diabetes.head(10)


# ## Conclusion 2
# 
# By looking at the columns SkinThickness,BloodPressure,Glucose and BMI, we come at a conclusion that their values can't be zeros. It means null values are represented by 0 in these columns.

# In[ ]:


def check_for_zero(columns):
    for col in columns:
        if 0 in diabetes[col]:
            print(col+' has 0 in it.')

columns = ['Glucose', 'BloodPressure', 'SkinThickness',
           'BMI', 'DiabetesPedigreeFunction', 'Age',]
check_for_zero(columns)


# These all column has zeros in it and it can't be, so we will find a way to replace these zeros:
# 1. One obvious way is to replace them with the mean, median or mode of respective columns, since they are all non categorical columns
# 2. Other way can be to replace with the mean, median or mode of respective columns based on outcome columns.
# 
# In this kernel we will follow first.

# In[ ]:


#before doing so, we will split our X and y
X = diabetes.drop('Outcome',axis=1)
Y = diabetes['Outcome']


# In[ ]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=0, strategy='mean')
X[columns] = imp.fit_transform(X[columns]) #fit the imputer


# In[ ]:


X.head(10)


# ## Conclusion 3
# Till now we have transformed our coulumns containing 0 in it. Now there is a column named Insulin which also have zeros but since I have no idea whether this column can contains zero too, so I will drop this column

# In[ ]:


X = X.drop('Insulin',axis=1)


# In[ ]:


#let's do some EDA
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(X['Pregnancies'])
X['Pregnancies'].value_counts()


# In[ ]:


def draw_dist(column):
    plt.figure()
    return sns.distplot(X[col])


# In[ ]:


for col in columns:
    draw_dist(col)


# In[ ]:


#Let us have some visualization about y
sns.countplot(Y)


# In[ ]:


#since all columns are not on same scale so let us normalize them
# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train,y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy on training data: {}'.format(knn_scaled.score(X_train,y_train)))
print('Accuracy on test data: {}'.format(knn_scaled.score(X_test,y_test)))


# In[ ]:


#implement logistic regression
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', LogisticRegression())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)

# Fit the pipeline to the training set: knn_scaled
logreg_scaled = pipeline.fit(X_train,y_train)

# Compute and print metrics
print('Accuracy on training data: {}'.format(logreg_scaled.score(X_train,y_train)))
print('Accuracy on test data: {}'.format(logreg_scaled.score(X_test,y_test)))


# In[ ]:


#implement logistic regression
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', SVC())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)

# Fit the pipeline to the training set: knn_scaled
Svm_scaled = pipeline.fit(X_train,y_train)

# Compute and print metrics
print('Accuracy on training data: {}'.format(Svm_scaled.score(X_train,y_train)))
print('Accuracy on test data: {}'.format(Svm_scaled.score(X_test,y_test)))


# In[ ]:




