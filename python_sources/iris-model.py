#!/usr/bin/env python
# coding: utf-8

# ***Iris Flower Model*** 
# 
# In this kernal I am going to explore following
# * Basic ploting
# * different classification algorithems
# * Cross_validation
# * GridSearchCV
# * RandamizedSearchCV
# * Converting categorical variable (different approaches)
# 

# > **Import required libraries using import**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # using for plots
import seaborn as sns #using for plots
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split # split train and test sets
from sklearn.preprocessing import StandardScaler # for scaling 

from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Random Forest
from sklearn.ensemble import RandomForestClassifier
# Gradient Boosting Machine
from sklearn.ensemble import GradientBoostingClassifier
# Cross Validation Score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.


# **Reading Iris flower dataset using pandas read_csv menthod**

# In[ ]:


iris_ds= pd.read_csv("../input/Iris.csv")
iris_ds.columns,iris_ds.shape
#print(iris_ds.shape)


# **Dataset contains 6 columns and 150 observations.**
# 
# Checking top 5 rows of the dataset using head. We can show number of rows based on parameter. By default it shows 5 rows.

# In[ ]:


iris_ds.head()


# checking last 5 rows using tail. By default tail shows last 5 rows.

# In[ ]:


iris_ds.tail()


# Check any null values in the datase using isnull and sum. Out data set not have any null/empty values.

# In[ ]:


iris_ds.isnull().sum()


# checking column data types using dtypes. Our dataset contains one int, 4 float and 1 object variables. 'Species' is our target varible.

# In[ ]:


iris_ds.dtypes


# We have total 6 columns in the data set. Id column we can delete. 'Species' is the catagorical column
# 
# Let check distribution of target vriable uisng value_counts
# 
# 

# In[ ]:


iris_ds.Species.value_counts()


# We have equal distribution values in Target variable. This is balenced dataset. In case of imbalenced datasets we have to use other techniques to handle. 
# 
# I am deleting Id column. This wont help our model. using drop function I am deleting Id variable from dataset. 
# 
# axis=1 indicated column level and inplace=True applies dataset.

# In[ ]:


iris_ds.drop(columns=['Id'],axis=1,inplace=True)


# Id column has been deleted from dataset. Now you can check how many columns present in the dataset

# In[ ]:


iris_ds.columns.values


# using hist function from pandas we can check numarical column distribution.

# In[ ]:


iris_ds.hist(figsize=(20,10))


# Using seaboarns package we can show rich plots. I used pairplot function to show scatter plots.

# In[ ]:


sns.pairplot(iris_ds,hue='Species')


# Box plot for all numarical variables

# In[ ]:


sns.boxplot(data=iris_ds)


# Using describe we can see min/max/std/quartail distribution for all numarical columns

# In[ ]:


iris_ds.describe()


# Info shows more info like how many observations there in the dataset
# 
# It is also very powerfull. It show how many missing values also in each column level.
# 
# It shows column data type also. Lot information with single command

# In[ ]:


iris_ds.info()


# Now I am going to convert target variable as category.
# we can follow different approaches for this convertion. different approaches as follows
# * LableEncoder()
# * map (We use map for ordinal category columns. You can specify the order)
# * type conversion using astype

# In[ ]:


iris_ds.Species = iris_ds.Species.astype('category')


# Now we check again data types using info command. now it shows type as category

# In[ ]:


iris_ds.info()


# We can access converted values using codes property.

# In[ ]:


iris_ds.Species.cat.codes.head()


# Placing converted values back to Species column.

# In[ ]:


iris_ds.Species = iris_ds.Species.cat.codes


# In[ ]:


iris_ds.Species.tail()


# Displaying columns in the dataset using columns property

# In[ ]:


iris_ds.columns.values


# Now I am copying target variable to y and all remaining columns to X variable. without this also you can pass directly to train_test_split method. 

# In[ ]:


X = iris_ds[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_ds.Species


# Split dataset for train and test. We use train for training purpose and test for validation.

# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
x_train.shape[0],y_train.shape[0]


# Using StandardScalar function scale all numarical variables. If we have any categorival variables we use OneHotEncoder() for create dummy variables. We can use pandas get_dummies also for to create dummy variables.

# In[ ]:


scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)


# In[ ]:


x_train


# I written normal_prediction() fuction for predicting output using different algorithems. 
# Following algorithems I used for predict
# * LogisticRegression
# * SVM
# * KNN
# * DecisionTree
# * ReandomForest
# * GradientBoosting

# In[ ]:


def normal_prediction():
    logis = LogisticRegression()
    logis.fit(x_train,y_train)
    print("logistic regression::\n",confusion_matrix(y_test,logis.predict(x_test)),"\n")
    
    svm = SVC()
    svm.fit(x_train,y_train)
    print("SVM ::\n",confusion_matrix(y_test,logis.predict(x_test)),"\n")
    
    knn = KNeighborsClassifier()
    knn.fit(x_train,y_train)
    print("KNN ::\n",confusion_matrix(y_test,knn.predict(x_test)),"\n")
    
    dTmodel = DecisionTreeClassifier()
    dTmodel.fit(x_train,y_train)
    print("DecisionTree ::\n",confusion_matrix(y_test,dTmodel.predict(x_test)),"\n")
    
    rForest = RandomForestClassifier()
    rForest.fit(x_train,y_train)
    print("RandomForest ::\n",confusion_matrix(y_test,rForest.predict(x_test)),"\n")

    grBoosting = GradientBoostingClassifier()
    grBoosting.fit(x_train,y_train)
    print("GradientBoosting ::\n",confusion_matrix(y_test,grBoosting.predict(x_test)),"\n")


# Calling normal_prediction() function

# In[ ]:


normal_prediction()


# Using Cross_val_score() function to predict output. This way we can use KFold cross validation. Here I used cv=5. This creates 5 folds.

# In[ ]:


#using cross_val_score
logis = LogisticRegression()
svm = SVC()
knn = KNeighborsClassifier()
dTmodel = DecisionTreeClassifier()
rForest = RandomForestClassifier()
grBoosting = GradientBoostingClassifier()
    
scores = cross_val_score(logis,x_train,y_train,cv=5)
print("Accuracy for logistic regresion: mean: {0:.2f} 2sd: {1:.2f}".format(scores.mean(),scores.std() * 2))
print("Scores::",scores)
print("\n")

scores2 = cross_val_score(svm,x_train,y_train,cv=5)
print("Accuracy for SVM: mean: {0:.2f} 2sd: {1:.2f}".format(scores2.mean(),scores2.std() * 2))
print("Scores::",scores)
print("\n")

scores3 = cross_val_score(knn,x_train,y_train,cv=5)
print("Accuracy for KNN: mean: {0:.2f} 2sd: {1:.2f}".format(scores3.mean(),scores3.std() * 2))
print("Scores::",scores)
print("\n")

scores4 = cross_val_score(dTmodel,x_train,y_train,cv=5)
print("Accuracy for Decision Tree: mean: {0:.2f} 2sd: {1:.2f}".format(scores4.mean(),scores4.std() * 2))
print("Scores::",scores4)
print("\n")

scores5 = cross_val_score(rForest,x_train,y_train,cv=5)
print("Accuracy for Random Forest: mean: {0:.2f} 2sd: {1:.2f}".format(scores5.mean(),scores5.std() * 2))
print("Scores::",scores5)
print("\n")

scores6 = cross_val_score(grBoosting,x_train,y_train,cv=5)
print("Accuracy for Gradient Boosting: mean: {0:.2f} 2sd: {1:.2f}".format(scores6.mean(),scores6.std() * 2))
print("Scores::",scores6)
print("\n")


# GridSearchCV and RandomizedSearchCV are ways to tune hyper parameters.

# In[ ]:


clf = RandomForestClassifier()
#Random Forest
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 4),
              "min_samples_split": sp_randint(2, 4),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 5
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)

random_search.fit(x_train, y_train)
print(random_search.best_params_)
print(random_search.best_estimator_)
confusion_matrix(y_test,random_search.predict(x_test))


# Using GridSearchCV to tune parameters

# In[ ]:


# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 4],
              "min_samples_split": [2, 3, 4],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)

grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
confusion_matrix(y_test,grid_search.predict(x_test))


# In[ ]:




