#!/usr/bin/env python
# coding: utf-8

# ![](http://)# Introduction 
# The sinking of Titanic is one of the most notorious shipwreck on the history. In 1912 during her voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew.
# 
# <font color = 'purple'>
# 
# Content:
# 
# 1. [Load and Check Data](#1)
# 2. [Variable Description](#2)
#      *  [Univariate Variable Analysis](#3)
#           * [Categorical Variable](#4)
#           * [Numerical Variable](#5)    
# 2. [Basic Data Analysis](#6)
# 2. [Outlier Detection](#7)
# 2. [Missing Value](#8)
#      *  [Find Missing Value](#9)
#      *  [Fill Missing Value](#10)
# 2. [Visualization](#11)
#      *  [Correlation Between Sibsp--Parch--Age--Fare--Survived](#12)
#      *  [SibSp--Survived](#13)
#      *  [Parch--Survived](#14)
#      *  [Pclass--Survived](#15)
#      *  [Age--Survived](#16)
#      *  [Pclass--Survived--Age](#17)     
#      *  [Embarked--Sex--Pclass--Survived](#18)
#      *  [Embarked--Sex--Fare--Survived](#19)
#      *  [Fill missing: Age Feature](#20)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import seaborn as sns
from collections import Counter 
import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <a id="1"></a>
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


# <a id="2"></a>
# # Variable Description
# 1. PassengerId : unique id number to each passenger
# 1. Survived : passenger survive (1) or died (0)
# 1. Pclass : passenger class
# 1. Name : name
# 1. Sex : gender of passenger
# 1. Age : age of passenger
# 1. SibSp : number of siblings or spouses
# 1. Parch : number of parents or children
# 1. Ticket : ticket number
# 1. Fare : amount of money spent on ticket
# 1. Cabin : cabin category
# 1. Embarked : port where passenger embarked (C=Cherbourg,Q=Queenstown,S=Southampton)

# In[ ]:


train_df.info()


# * float64(2) : Fare and Age
# * int64(5) : PassengerId,Survived,Pclass,SibSp and Parch
# * object(5) : Name, Sex, Ticket,Cabin and Embarked

# <a id="3"></a>
# # Univariate Variable Analysis
# 
# * Categorical Variable : Survived, Pclass,Sex,SibSp,Parch,Cabin,Embarked, Name and Ticket
# * Numerical Variable : Age, PassengerId and Fare
# 

# <a id="4"></a>
# ### Categorical Variable

# In[ ]:


def bar_plot(variable):
    var=train_df[variable]
    varValue=var.value_counts()
    
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: /n {}".format(variable,varValue))
    
    


# In[ ]:


category1=["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:
    bar_plot(c)


# In[ ]:


category2=["Cabin","Name","Ticket"]
for c in category2:
    print("{}: /n".format(train_df[c].value_counts()))


# <a id="5"></a>
# ## Numerical Variable

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable])
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[ ]:


numericVar=["Fare","Age","PassengerId"]
for n in numericVar:
    plot_hist(n)


# <a id="6"></a>
# # Basic Data Analysis
# * Pclass-Survived
# * Sex-Survived
# * SibSp-Survived
# * Parch-Survived

# In[ ]:


#Pclass-Survived

train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


#Sex-Survived

train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


#SibSp-Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


#Parch-Survived
train_df[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# <a id="7"></a>
# # Outlier Detection

# In[ ]:


def detect_outliers(df,features):
    outlier_indices=[]
    
    for c in features:
        #1st quartile
        Q1=np.percentile(df[c],25)
        #3rd quartile
        Q3=np.percentile(df[c],75)
        #IQR
        IQR=Q3-Q1
        #Outlier step
        outlier_step=IQR*1.5
        #detect outlier and their indices
        outlier_list_col=df[(df[c]<Q1-outlier_step) | (df[c]<Q3+ outlier_step)].index
        #store indeces
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices=Counter(outlier_indices) 
    multiple_outliers=list(i for i, v in outlier_indices.items() if v>2)
    
    return multiple_outliers


# In[ ]:


train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]


# In[ ]:


train_df=train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]), axis=0).reset_index(drop=True)


# <a id="8"></a>
# # Missing Value
# *      Find Missing Value
# *      Fill Missing Value
# 

# In[ ]:


train_df_len=len(train_df)
train_df=pd.concat([train_df,test_df],axis=0).reset_index(drop=True)


# In[ ]:


train_df.head()


# ## Find Missing Value

# In[ ]:


train_df.columns[train_df.isnull().any()]


# In[ ]:


train_df.isnull().sum()


# <a id="10"></a>
# ## Fill Missing Value
# * Embarked has 2 missing value
# * Fare only has 1

# In[ ]:


train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df.boxplot(column="Fare", by="Embarked")
plt.show()


# In[ ]:


train_df["Embarked"]=train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df[train_df["Fare"].isnull()]


# In[ ]:



train_df["Fare"]= train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))


# <a id="11"></a>
# # Visualization

# <a id="12"></a>
# ## Correlation Between Sibsp--Parch--Age--Fare--Survived

# In[ ]:


list1=["SibSp","Parch","Age","Fare","Survived"]
sns.heatmap(train_df[list1].corr(), annot=True ,fmt=".2f")
plt.show()


# Fare feature seems to have correlation with survived feature (0.26)

# <a id="13"></a>
# ## SibSp--Survived

# In[ ]:


g=sns.factorplot(x="SibSp", y="Survived", data=train_df , kind="bar" ,size=6)
g.set_ylabels("Survived Probability")
plt.show()


# * Having a lot of SibSp decreases the chance to survive.
# * If SibSp== 0 or 1 or 2 passenger has more chance to survive. 
# * We can consider a new feature describing these categories.

# <a id="14"></a>
# ## Parch--Survived

# In[ ]:


g=sns.factorplot(x="Parch", y="Survived", data=train_df , kind="bar" ,size=6)
g.set_ylabels("Survived Probability")
plt.show()


# * SibSp and Parch can be used for new feature extraction with th=3
# * Small families have more chance to survived.
# * There is standard deviation in survival of passenger with parch=3

# <a id="15"></a>
# ## Pclass--Survived

# In[ ]:


g=sns.factorplot(x="Pclass", y="Survived", data=train_df , kind="bar" ,size=6)
g.set_ylabels("Survived Probability")
plt.show()


# <a id="16"></a>
# ## Age--Survived

# In[ ]:


g= sns.FacetGrid(train_df, col="Survived")
g.map(sns.distplot, "Age", bins=25)
plt.show()


# * Age<=10 has a high survival rate
# * Oldest passenger (80) survived
# * Large number of 20 years old did not survived.
# * Most passengers 15-35 age range
# * Use age feature in training
# * Use age distribution for missing value of age
# 

# <a id="17"></a>
# ## Pclass--Survived--Age

# In[ ]:


g= sns.FacetGrid(train_df, col="Survived", row= "Pclass")
g.map(plt.hist, "Age", bins=25)
g.add_legend()
plt.show()


# * Pclass is a important feature for model training

# <a id="18"></a>
# ## Embarked--Sex--Pclass--Survived

# In[ ]:


g= sns.FacetGrid(train_df, row= "Embarked" , size=2)
g.map(sns.pointplot, "Pclass","Survived","Sex")
g.add_legend()
plt.show()


# * Female passengers have much better survival rate than males.
# * Males better survival rate in Pclass 3 in C.
# * Embarked and sex will be used at training

# <a id="19"></a>
# ## Embarkes--Sex--Fare--Survived

# In[ ]:


g= sns.FacetGrid(train_df, col="Survived", row= "Embarked" , size=2.3)
g.map(sns.barplot,"Sex", "Fare")
g.add_legend()
plt.show()


# * Passengers who pay higher fare have better survival. Fare can be used categorical for training.

# <a id="20"></a>
# ## Fill missing: Age Feature

# In[ ]:


train_df[train_df["Age"].isnull()]


# In[ ]:


sns.factorplot(x="Sex", y="Age", data= train_df, kind="box")
plt.show()


# Sex is not informative for age prediction.Age distribution, median seem to be same both for female and male.

# In[ ]:


sns.factorplot(x="Sex", y="Age", hue="Pclass", data= train_df, kind="box")
plt.show()


# First class passengers are older than 2nd and 2nd class passengers are older than 3rd class passengers.

# In[ ]:


sns.factorplot(x="Parch", y="Age", data= train_df, kind="box")
sns.factorplot(x="SibSp", y="Age", data= train_df, kind="box")
plt.show()


# In[ ]:


train_df["Sex"]=[1 if i=="male" else 0 for i in train_df["Sex"]]


# In[ ]:


sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot=True)
plt.show()


# Age is not correlated with sex but it is correlated with Parch,Sibsp,Pclass

# In[ ]:


index_nan_age=list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred=train_df["Age"][((train_df["SibSp"]==train_df.iloc[i]["SibSp"])& (train_df["Parch"]==train_df.iloc[i]["Parch"])&(train_df["Pclass"]==train_df.iloc[i]["Pclass"]))].median()
    age_med=train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i]=age_pred
    else:
        train_df["Age"].iloc[i]=age_med


# In[ ]:


train_df[train_df["Age"].isnull()]


# In[ ]:





# In[ ]:




