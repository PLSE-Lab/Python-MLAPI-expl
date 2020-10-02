#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The RMS Titanic sank in the early morning hours of 15 April 1912 in the North Atlantic Ocean, four days into the ship's maiden voyage from Southampton to New York City. The largest ocean liner in service at the time, Titanic had an estimated 2,224 people on board when she struck an iceberg at around 23:40 on Sunday, 14 April 1912. Her sinking two hours and forty minutes later at 02:20 on Monday, 15 April, resulted in the deaths of more than 1,500 people, making it one of the deadliest peacetime marine disasters in history.
# 
# <font color = "blue">
# 
# Content
#     
# 1. [Load and Check Data](#1)
# 1. [Variable Description](#2)
#     * [Univariate Variable Analysis](#3)
#         * [Categorical Variable](#4)
#         * [Numerical Variable](#5)
# 1. [Basic Data Analysis](#6)
# 1. [Outlier Detection](#7)
# 1. [Missing Value](#8)
#     * [Find Missing Value](#9)
#     * [Fill Missing Value](#10)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id = "1"></a>
# # Load and check Data

# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId = test_df['PassengerId']


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# <a id = "2"></a>
# # Variable Description

# 1. PassengerId: Unique to each passenger
# 1. Survived: passenger survive(1) or died(0) 
# 1. Pclass: passenger class
# 1. Name: name of passenger 
# 1. Sex: gender of passenger 
# 1. Age: age of passenger 
# 1. SibSp: number of siblings/spouses
# 1. Parch: number of parents/children
# 1. Ticket: ticket number
# 1. Fare: amount of money spent on ticket
# 1. Cabin: cabin category 
# 1. Embarked: port where passenger embarked (C = Cherbourg, Q = Queenstown, S = Southampton)

# In[ ]:


train_df.info()


# * float64(2): Age, Fare
# * int64(5): PassengerId, Survived, Pclass, SibSp, Parch
# * object(5): Name, Sex, Ticket, Cabin, Embarked

# <a id = "3"></a>
# # Univariate Variable Analysis
# 
#    * Categorical Variable: Survived, Pclass, SibSp, Parch, Name, Sex, Ticket, Cabin, Embarked
#    * Numerical Variable: PassengerId, Age, Fare

# <a id = "4"></a>
# ## Categorical Variable

# In[ ]:


def bar_plot(variable):

    # get feature
    var = train_df[variable]
    # count number of categorical variable
    varValue = var.value_counts()
    
    # visualization
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{} \n {}".format(variable,varValue))


# In[ ]:


category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]
for c in category1:
    bar_plot(c)


# <a id = "5"></a>
# ## Numerical Variable

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[ ]:


numericalVar = ["Fare","Age","PassengerId"]
for n in numericalVar:
    plot_hist(n)


# <a id = "6"></a>
# # Basic Data Analysis
# 
# * Pclass - Survived
# * Sex - Survived
# * SibSp - Survived
# * Parch - Survived
#       
# 
#     

# In[ ]:


# Pclass vs Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by = "Survived", ascending = False)


# In[ ]:


# Sex vs Survived
train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by = "Survived", ascending = False)


# In[ ]:


# SibSp vs Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by = "Survived", ascending = False)


# In[ ]:


# Parch vs Survived
train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by = "Survived", ascending = False)


# <a id = "7"></a>
# # Outlier Detection

# In[ ]:


def detect_outliers(df,features):
    outlier_indices = []
    for c in features:
        #1st Quartile
        Q1 = np.percentile(df[c],25)
        #3rd Quartile
        Q3 = np.percentile(df[c],75)
        #IQR
        IQR = Q3 - Q1
        #outlier step:
        outlier_step = IQR * 1,5
        #detect outlier and their indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        #store indices
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


# In[ ]:


train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]


# In[ ]:


#drop outliers
train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]), axis = 0).reset_index(drop = True)


# <a id = "8"></a>
# # Missing Value
# * Find Missing Value
# * Fill Missing Value

# In[ ]:


train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df], axis = 0).reset_index(drop = True)


# In[ ]:


train_df.head()


# <a id = "9"></a>
# ## Find Missing Value

# In[ ]:


train_df.columns[train_df.isnull().any()]


# In[ ]:


train_df.isnull().sum()


# <a id = "10"></a>
# ## Fill Missing Value

# In[ ]:


train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df.boxplot(column = "Fare", by = "Embarked")
plt.show()


# In[ ]:


train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df.isnull().sum()

