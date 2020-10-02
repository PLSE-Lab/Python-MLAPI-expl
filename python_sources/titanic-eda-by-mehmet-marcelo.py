#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# The sinking of Titanic is one of the most infamous shipwrecks in the history. In 1912, during her first voyage, the Titanic sank after colliding with an iceberg. After this shipwreck, 1502 out of 3547 passengers and crew killed. Also, interest in this shipwreck has been gradually increasing.
# 
# <font color="blue">
# Content:
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


# <a id="1"></a><br>
# # Load and Check Data

# In[ ]:


train_df=pd.read_csv("/kaggle/input/titanic/train.csv")
test_df=pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId=test_df["PassengerId"]


# In[ ]:


train_df.columns 


# In[ ]:


len(train_df.columns)


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# <a id = "2"></a><br>
# # Variable Description

# 1. PassengerId: unique ID number to each passenger
# 1. Survived: Passenger survive(1) or died(0) 
# 1. Pclass: Passenger class
# 1. Name name 
# 1. Sex: Gender of passenger
# 1. Age: Age of passenger
# 1. SibSp: Number of siblings/spouses
# 1. Parch: Number of parents/children
# 1. Ticket: Ticket number
# 1. Fare: Amount of money spent on ticket
# 1. Cabin: Cabin category
# 1. Embarked: Port where passenger embarked(C=Cherbourg, Q=Queenstown, S=Southampton)

# In[ ]:


train_df.info()


# * float64(2): Age and Fare
# * int64(5): PassengerId, Survived, Pclass, SibSp and Parch
# * object(5): Cabin and Embarked

# <a id="3"></a><br>
# # Univariate Variable Analysis
# * Categorical Variable: Survived, Sex, Pclass, Cabin, Name, Ticket, SibSp and Parch
# * Numeric Variable: Agei PassengerId and Fair

# <a id="4"></a><br>
# ## Categorical Variable

# In[ ]:


def bar_plot(variable):
    """
        input: variable ex: "Sex"
        output: bar plot & value count
    """
    # get feature
    var = train_df[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n{}".format(variable,varValue))


# In[ ]:


category1= ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]
for c in category1:
    bar_plot(c)


# In[ ]:


category2= ["Cabin","Name","Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))


# <a id="5"></a><br>
# ## Numerical Variable

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable],bins=50)
    plt.xlabel(variable)
    plt.ylabel("{} distribution with histogram".format(variable))
    plt.show()


# In[ ]:


numericVar=["Fare","Age","PassengerId"]
for n in numericVar:
    plot_hist(n)


# <a id="6"></a><br>
# # Basic Data Analysis

# * Pclass - Survived
# * Sex - Survived
# * SibSp - Survived
# * Parch - Survived

# In[ ]:


train_df[["Pclass","Survived"]]


# In[ ]:


train_df[["Pclass","Survived"]].groupby(["Pclass"]).mean()
#if we do not use as_index=False,


# In[ ]:


#Pclass vs Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


#Sex vs Survived
train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


#Sibsp vs Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


#Parch vs Survived
train_df[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


#Embarked vs Survived
train_df[["Embarked","Survived"]].groupby(["Embarked"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


#Age vs Survived
train_df[["Age","Survived"]].groupby(["Age"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# <a id="7"></a><br>
# # Outlier Detection

# In[ ]:


def detect_outliers(df, features):
    outlier_indices = []
    
    for c in features:
        #1st quartile
        Q1 = np.percentile(df[c],25)
        #3rd quartile
        Q3 = np.percentile(df[c],75)
        #IQR
        IQR = Q3 - Q1
        #Outlier step
        outlier_step = IQR * 1.5
        #Detect outlier and indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step)| (df[c] > Q3 + outlier_step)].index
        #store indices
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices=Counter(outlier_indices)
    #if there are more than two outliers in the same sample, we will remove it from list:
    multiple_outliers=list(i for i,v in outlier_indices.items() if v>2) 
    return multiple_outliers


# In[ ]:


train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]


# <a id="8"></a><br>
# # Missing Value
# * Find Missing Value
# * Fill Missing Value

# In[ ]:


train_df_len=len(train_df)
train_df=pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
#if your run this cell more one time, an error occurs here.


# In[ ]:


train_df.head()


# <a id="9"></a><br>
# ## Find Missing Value

# In[ ]:


train_df.columns[train_df.isnull().any()]
#which columns have missing values


# In[ ]:


train_df.isnull().sum()
#total number of missing values in columns 


# <a id="10"></a><br>
# ## Fill Missing Value
# * Embarked has 2 missing values
# * Fare has only 1 missing value

# In[ ]:


train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df.boxplot(column="Fare", by="Embarked",figsize=(11,11))
plt.show()


# In[ ]:


train_df["Embarked"]=train_df["Embarked"].fillna("C")


# In[ ]:


train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df["Fare"]=train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))


# In[ ]:


train_df[train_df["Fare"].isnull()]

