#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The sinking of Titanic is a shipwreck which is known by all world.In 1912,during her voyage,The Titanic sank  after colliding with an iceberg killing 1502 out of 2224 passengers and crew.Our job is handle Titanic Dataset.
# 
# <font color = 'turquoise'>
# Content:
# <font color = 'turquoise'>
# 1. [Load and Check Data](#1) 
#     <font color = 'turquoise'>    
# 1. [Variable Description](#2)
#     * [Univariate Variable Analysis](#3)
#         * [Categorical Variable Analysis](#4)
#         * [Numerical Variable Analysis](#5)
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
plt.style.use("seaborn-whitegrid") #it is forgraphics,gives grid view

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


# <a id = "1"></a><br>
# # Load and Check Data
# 

# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"]


# <a id = "2"></a><br>
# # Variable Description
# 1. PassengerId: Unique id number to each passenger
# 1. Survived: Passenger survived(1) or died(0)
# 1. Pclass: Passenger class
# 1. Name
# 1. Sex
# 1. Age
# 1. SibSp: Number of siblings/spouses
# 1. Parch: Number of parents/children
# 1. Ticket: Ticket number
# 1. Fare: Ticket price
# 1. Cabin: Cabin category
# 1. Embarked: Boarding Port(C = Cherbourg, Q = Queenstown, S = Southampton)
# 
# 

# In[ ]:


train_df.info()


# * float64(2) : Fare and Age
# * int64(5): Pclass,Sibsp,Parch,PassengerId and Survived
# * object(5): Cabin,Embarked,Ticket,Name and Sex

# * [Univariate Variable Analysis](#3)
#     * [Categorical Variable](#4)
#     * [Numerical Variable](#5)

# <a id = "3"></a><br>
# # Univariate Variable Analysis
# * Categorical Variable: Survived,Sex,Pclass,Embarked,Cabin,Name,Ticket,Sibsp and Parch
# * Numerical Variable: Fare,Age and PassengerId

# <a id = "4"></a><br>
# ## Categorical Variable

# In[ ]:


def bar_plot(variable):
    """
        input variable ex: "Sex"
        output: bar plot & value count
    """
    #get feature
    var = train_df[variable]
    #count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))


# In[ ]:


category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp","Parch"]
for c in category1:
    bar_plot(c)


# In[ ]:


category2 = ["Cabin", "Name", "Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))


# <a id = "5"></a><br>
# ## Numerical Variable

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable],bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[ ]:


numericVar = ["Fare", "Age", "PassengerId"]
for n in numericVar:
    plot_hist(n)


# <a id = "6"></a><br>
# # Basic Data Analysis
# * Pclass - Survived
# * Sex - Survived
# * SibSp - Survived
# * Parch - Survived

# In[ ]:


#Pclass vs Survived
#Effect of class on surviving
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by = "Survived",ascending = False)


# In[ ]:


#Sex vs Survived
#Effect of gender on surviving
train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by = "Survived",ascending = False)


# In[ ]:


#SibSp vs Survived
#Effect of having siblings or spouses on surviving
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by = "Survived",ascending = False)


# In[ ]:


#Parch vs Survived
#Effect of having parents or children on surviving
train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by = "Survived",ascending = False)


# <a id = "7"></a><br>
# # Outlier Detection

# In[ ]:


def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        #1st quartile
        Q1 = np.percentile(df[c],25)
        #3rd quartile
        Q3 = np.percentile(df[c],75)
        #IQR
        IQR = Q3-Q1
        #Outlier step
        outlier_step = IQR * 1.5
        #Detect outlier and their indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        #Store indices
        outlier_indices.extend(outlier_list_col)
    #Counting numbers of each element
    outlier_indices = Counter(outlier_indices)
    #Take if there are more than 1 outliers,not just 1 outlier
    multiple_outliers = list(i for i,v in outlier_indices.items() if v > 2)
    return multiple_outliers


# In[ ]:


train_df.loc[detect_outliers(train_df, ["Age","SibSp","Parch","Fare"])]


# In[ ]:


#Drop outliers
train_df = train_df.drop(detect_outliers(train_df, ["Age","SibSp","Parch","Fare"]), axis = 0).reset_index(drop = True)


# <a id = "8"></a><br>
# # Missing Value
# * Find Missing Value
# * Fill Missing Value
# 

# In[ ]:


train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)


# <a id = "9"></a><br>
# ## Find Missing Value

# In[ ]:


train_df.columns[train_df.isnull().any()]


# <a id = "10"></a><br>
# ## Fill Missing Value
# * Embarked has 2 missing value
# * Fare has only 1 missing value

# In[ ]:


train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df.boxplot(column = "Fare",by = "Embarked")
plt.show()


# It seems that these 2 passengers got into the Titanic from Cherbourg Port so I will fill Embarked values with C.

# In[ ]:


train_df["Embarked"] = train_df["Embarked"].fillna("C")


# In[ ]:


train_df[train_df["Fare"].isnull()]


# I'm going to fill Fare value with the average of 3rd class prices because Mr.Thomas has a 3rd class ticket.

# In[ ]:


train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))

