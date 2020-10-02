#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Titanic is well-known shipwreck happened ind 1912. The Titanic sank because of hitting to iceberg and leads to death 1502 people out of 2224 passengers and crew.
# 
# <font color = 'blue'>
# Content:
# 
# 1. [Load and Check Data](#1)
# 2. [Variable Description](#2)
#     * [Univariate Variable Analysis](#3)
#         * [Categorical Variable Analysis](#4)
#         * [Numerical Variable Analysis](#5)
# 3. [Basic Data Analysis](#6)
# 4. [Outlier Detection](#7)
# 5. [Missing Value](#8)
#     * [Find Missing Value](#9)
#     * [Fill Missing Value](#10)
# 1. [Visualization](#11)
#     * [Correlation Between Sibsp -- Parch -- Age -- Fare -- Survived](#12)
#     * [SibSp -- Survived](#13)
#     * [Parch -- Survived](#14)
#     * [Pclass -- Survived](#15)
#     * [Age -- Survived](#16)
#     * [Pclass -- Survived -- Age](#17)
#     * [Embarked -- Sex -- Pclass -- Survived](#18)
#     * [Embarked -- Sex -- Fare -- Survived](#19)
#     * [Fill Missing: Age Feature](#20)

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


# <a id = '1'></a><br>
# # Load and Check Data

# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"]


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# <a id = '2'></a><br>
# # Variable Description
# 1. PassengerId : Unique id number for each passenger
# 1. Survived : passenger survived(1) or died(0)
# 1. Pclass : Passenger class
# 1. Name : Name
# 1. Sex : Gender of passengers Male(1) and Female(0)
# 1. Age : Age of passengers
# 1. SibSp : nubmer of siblings/spouses
# 1. Parch  : number of parents/children
# 1. Ticket : Ticket number
# 1. Fare : Amount of money spent for ticket
# 1. Cabin : Cabin category
# 1. Embarked : Port where passenger embarked (C = Cherbourg, Q = Queenstown, S = Southampton)

# In[ ]:


train_df.info()


# * float64(2) : Fare and Age
# * int64(5) : Pclass, sibsp, parch, passengerId and survived
# * object(5): Cabin, embarked, ticket, name and sex

# <a id = '3'></a><br>
# # Univariate Variable Analysis
# * Categorical Variable Analysis : Survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, Sibsp and Parch
# * Numerical Variable Analysis : Fare, Age and passengerId

# <a id = '4'></a><br>
# ## Categorical Variable

# In[ ]:


def bar_plot(variable):
    """
    input: variable ex: "Sex"
    output: bar plot & value count
    
    """
    # get feature
    var = train_df[variable]
    #counts number of categorical variable (value/sample)
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize = (10,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
    


# In[ ]:


category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]
for c in category1:
    bar_plot(c)


# <a id = '5'></a><br>
# ## Numerical Variable

# In[ ]:


def plot_hist(variable):
    """
    Fare, Age and passengerId
    """
    
    
    var = train_df[variable]
    
    #Visualize
    plt.figure(figsize = (10,3))
    plt.hist(var,bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} Distribution with histogram".format(variable))
    plt.show()
    


# In[ ]:


numericVar = ["Fare", "Age", "PassengerId"]
for c in numericVar:
    plot_hist(c)


# <a id = '6'></a><br>
# # Basic Data Analysis
# * Pclass - Survived
# * Sex - Survived
# * Sibsp - Survived
# * Parch - Survived

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


# In[ ]:


# Pclass,Sex vs Survived
train_df[["Pclass","Survived","Sex"]].groupby(["Pclass","Sex"], as_index = False).mean().sort_values(by = "Survived", ascending = False)


# In[ ]:


# Embarked vs Survived
train_df[["Embarked","Survived"]].groupby(["Embarked"], as_index = False).mean().sort_values(by = "Survived", ascending = False)


# In[ ]:


# Embarked,Pclass vs Survived 
train_df[["Embarked","Survived","Pclass"]].groupby(["Embarked","Pclass"], as_index = False).mean().sort_values(by = "Survived",ascending = False)


# <a id = '7'></a><br>
# # Outlier Detection

# In[ ]:


def detect_outliers(df,features):
    outlier_indices = []
    
    for i in features:
        #1st quartile
        
        Q1 = np.percentile(df[i],25)
        
        #3rd quartile
        
        Q3 = np.percentile(df[i],75)
        
        # IQR
        
        IQR = Q3 - Q1
        
        # outlier step
        
        outlier_step = IQR * 1.5
        
        #detect outlier and their indeces
        
        
        outlier_list_col = df[(df[i] < Q1 - outlier_step) | (df[i] > Q3 + outlier_step)].index
        
       
        
        
        #store indeces
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    
    multiple_outliers = list(i for i, v in outlier_indices.items() if v>2)
        
    return multiple_outliers
        


# In[ ]:


train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]


# In[ ]:


# drop outliers

train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)


# In[ ]:


a= train_df[(train_df["Age"] > 60) | (train_df["Fare"] >70)].index


# In[ ]:


a = Counter(a)
a


# <a id = '8'></a><br>
# # Missing Value
# * Find Missing Value
# * Fill Missing Value

# In[ ]:


train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df], axis = 0).reset_index(drop = True)


# In[ ]:


train_df.head()


# <a id = '9'></a><br>
# ## Find Missing Value

# In[ ]:


train_df.columns[train_df.isnull().any()]


# In[ ]:


train_df.isnull().sum()


# <a id = '10'></a><br>
# ## Fill Missing Value
# * Embarked has 2 missing value
# * Fare has only 1

# In[ ]:


train_df[train_df["Embarked"].isnull()]


# In[ ]:


trainEM_df = train_df[train_df.Pclass == 1]   #Filter for Pclass =1 due to our missing embarked datas have 1 Pclass
trainEM_df.boxplot(column = 'Fare', by="Embarked")
plt.show()


# Filling both missing Embarked datas with "C" would be good. Since they both paid 80.0 and C port has closer fares to 80.

# In[ ]:


# Filling Embarked

train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df[train_df["Fare"].isnull()]


# In[ ]:


train_df[(train_df.Embarked == "S") & (train_df.Pclass == 3)].boxplot(column = "Fare")
plt.show()


# In[ ]:


print(train_df[(train_df.Embarked == "S") & (train_df.Pclass == 3)]["Fare"].mean()) #Filled that empty fare with 13.642
train_df["Fare"] = train_df["Fare"].fillna(train_df[(train_df.Embarked == "S") & (train_df.Pclass == 3)]["Fare"].mean())
train_df[train_df["Fare"].isnull()]


# <a id = '11'></a><br>
# # Visualization

# <a id = '12'></a><br>
# ## Correlation Between SibSp -- Parch -- Age -- Fare -- Survived

# In[ ]:


train_df.head()


# In[ ]:


listem = ["SibSp", "Parch", "Age", "Fare", "Survived"]

plt.figure(figsize= (12,9))
sns.heatmap(train_df[listem].corr(), annot=True, fmt='.2f')
plt.show()


# Fare and Survived features have positive correlation as 0.26

# <a id = '13'></a><br>
# ## SibSp -- Survived

# In[ ]:


f,ax = plt.subplots(figsize = (10,10))

ax = sns.barplot(x='SibSp', y="Survived", data=train_df)

plt.xlabel("SibSp",fontsize=20)
plt.ylabel("Survival Probability",fontsize=20)
ax.tick_params(labelsize=20)
plt.show()


# * As we can see here, having more than 2 SibSp leads to sharp decrease of Survival Probabiltiy
# * If SibSp is 2 or less than 2, passenger has more chance to survive
# * We can consider a new feature describing these categories.

# <a id = '14'></a><br>
# ## Parch -- Survived

# In[ ]:


f,ax = plt.subplots(figsize = (10,10))

ax = sns.barplot(x='Parch', y="Survived", data=train_df)

plt.xlabel("Parch",fontsize=20)
plt.ylabel("Survival Probability",fontsize=20)
ax.tick_params(labelsize=20)
plt.show()


# * SibSp and Parch can be used for new feature creation with threshold=3.
# * Small families have more chance to survive.
# * There is std in survival of passenger with Parch=3

# <a id = '15'></a><br>
# ## Pclass -- Survived

# In[ ]:


f,ax = plt.subplots(figsize = (10,10))

ax = sns.barplot(x='Pclass', y="Survived", data=train_df)

plt.xlabel("SibSp",fontsize=20)
plt.ylabel("Survival Probability",fontsize=20)
ax.tick_params(labelsize=20)
plt.show()


# <a id = '16'></a><br>
# ## Age -- Survived

# In[ ]:




ax = sns.FacetGrid(train_df, col='Survived',size=5)

ax.map(sns.distplot, 'Age',bins=35)


# * age <=10 has a high survival rate,
# * oldest passenger (80) survived,
# * large number of 20 years old people did not survived,
# * most passengers are in 15-35 age range,
# * we can use age feature as training,
# * we can use age distribution for missing value of age feature

# <a id = '17'></a><br>
# ## Pclass -- Survived -- Age

# In[ ]:


g = sns.FacetGrid(train_df, col="Survived", row="Pclass", size=3)
g.map(plt.hist,"Age", bins=35)
g.add_legend()
plt.show()


# * Pclass is important for model training.

# <a id = '18'></a><br>
# ## Embarked -- Sex -- Pclass -- Survived

# In[ ]:


g = sns.FacetGrid(train_df, row="Embarked",size=3)
g.map(sns.pointplot, "Pclass","Survived","Sex")
g.add_legend()
plt.show()


# * Females have more survival rate than males
# * Males have more survival rate in pclass 3 in C.
# * Embarked and Sex will be used in training.

# <a id = '19'></a><br>
# ## Embarked -- Sex -- Fare -- Survived

# In[ ]:


g = sns.FacetGrid(train_df, row="Embarked", col='Survived')
g.map(sns.barplot, "Sex", "Fare")
plt.show()


# * Passsengers who pay higher fare have better survival. Fare can be used as categorical for training.

# <a id = '20'></a><br>
# ## Fill Missing: Age Feature

# In[ ]:


train_df[train_df["Age"].isnull()]


# In[ ]:


plt.subplots(figsize=(9,9))
sns.boxplot(data=train_df, x="Sex", y="Age")
plt.show()


# * Sex is not helping for us to predict age.

# In[ ]:


plt.subplots(figsize=(7,7))
sns.boxplot(data=train_df, x="Sex", y="Age",hue="Pclass")
plt.show()


# * 1st class passengers are older than 2nd, and 2nd is older than 3rd class.

# In[ ]:


plt.subplots(figsize=(9,9))
sns.boxplot(data=train_df, x="Parch", y="Age")
plt.show()


# In[ ]:


plt.subplots(figsize=(9,9))
sns.boxplot(data=train_df, x="SibSp", y="Age")
plt.show()


# In[ ]:


sns.heatmap(train_df[["Age","SibSp","Parch","Pclass"]].corr(), annot=True)
plt.show()


# * Age is correlated with Parch,Pclass,SibSp

# In[ ]:


index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)

for i in index_nan_age:

    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & (train_df["Parch"] == train_df.iloc[i]["Parch"])&  (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    
    age_med = train_df["Age"].median()

    if not np.isnan(age_pred):
         
        train_df["Age"].iloc[i] = age_pred
    else:
         train_df["Age"].iloc[i] = age_med


# In[ ]:


train_df[train_df["Age"].isnull()]


# In[ ]:




