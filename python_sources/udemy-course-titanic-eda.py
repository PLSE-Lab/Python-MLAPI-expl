#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <font color="#9B59B6 "/>
# 
#  RMS Titanic was a British passenger liner operated by the White Star Line that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters. 
#  [Information Source](https://en.wikipedia.org/wiki/RMS_Titanic) 
# 
# ![Titanic.gif](https://media1.tenor.com/images/3581859ed9fb6778648144db7542c451/tenor.gif)  
# 

# <font color="#99A3A4"/>
# Content: 
# 
# 
# <font color="#ECDD66"/>
# 
# 1. [Load and Check Data](#1)
# 2. [Variable Description](#2)
#     * [Univariate Variable Analysis](#3)
#         * [Categorical Variable Analysis](#4)
#         * [Numerical Variable Analysis](#5)
# 3. [Basic Data Analysis](#6)
# 4. [Outlier Detection](#7)
# 5. [Missing Value](#8)
#       * [ Find Missing Value](#9)
#       * [ Fill Missing Value](#10)

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


# <a id="1"></a><br>
# ## Load and Check Data

# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

train_df.columns


# In[ ]:


test_df.columns


# In[ ]:


test_PassengerId = test_df["PassengerId"]


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# <a id="2"></a><br>
# ## Variable Description
# <font color="#9B59B6 "/>
# 
# 1. PassengerId : Unique ID number for each passenger
# 2. Survived : Passenger which survived(1) or died(0)
# 3. Pclass : Passenger class
# 4. Name : Name
# 5. Sex : Gender of passengers
# 6. Age : Age of passengers
# 7. SibSp : Number of siblings/spouses
# 8. Parch : Number of parent/children
# 9. Ticket : Ticket number
# 10. Fare : Amount of money for spending on ticket
# 11. Cabin : Cabin categories
# 12. Embarked : Ports where passengers embarked (C: Cherborg, 
#  Q: Queenstown, S = Southampton)
# 
# 
# 

# In[ ]:


train_df.info()


# <font color="#9B59B6"/>
# 
# * float64(2) : Fare and Age
# * int64(5) : PassengerId, Sruvived, Pclass, SibSp, Parch        
# * object(5) : Name, Sex, Ticket, Cabin, Embarked

#  <a id="3"></a><br>
#  ## Univariate Variable Analysis
#  <font color="#9B59B6"/>
#  
#    * Categorical Variable Analysis : Survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, SibSp, Parch
#    * Numerical Variable Analysis : Age, Fare, PassengerId

#  <a id="4"></a><br>
# #### Categorical Variable Analysis

# In[ ]:


def bar_plot(variable):
    
    #Get feature
    var = train_df[variable]
    #Count number of categorical variable
    varValue = var.value_counts()
    
    #Visualize the data
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}:\n{}".format(variable,varValue))


# In[ ]:


category1 = ["Survived","Sex", "Pclass", "Embarked", "SibSp", "Parch"]
for c in category1:
    bar_plot(c)


# In[ ]:


category2 = ["Cabin", "Name", "Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))


#  <a id="5"></a><br>
# #### Numerical Variable Analysis

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable], bins=70)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[ ]:


numericVar = ["Fare", "Age", "PassengerId"]
for n in numericVar:
    plot_hist(n)
    


# <a id="6"></a><br>
# ## Basic Data Analysis
# <font color="#9B59B6"/>
# 
# * Pclass - Survived Correlation
# * Sex - Survived Correlation
# * SibSp - Survived Correlation
# * Parch - Survived Correlation 
# * Embarked - Survived Correlation
# * Embarked - Fare Correlation

# In[ ]:


#Pclass - Survived Correlation
corr1= train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values("Survived", ascending=False)
corr1


# In[ ]:


corr1.plot()
plt.show()


# In[ ]:


# Sex - Survived Correlation
corr2 = train_df[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values("Survived", ascending=False)
corr2


# In[ ]:


#SibSp - Survived Correlation
corr3 = train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values("Survived", ascending=False)
corr3


# In[ ]:


#Parch - Survived Correlation
corr4 = train_df[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean().sort_values("Survived", ascending=False)
corr4


# In[ ]:


# Embarked - Survived Correlation
corr5 = train_df[["Embarked", "Survived"]].groupby(["Embarked"], as_index = False).mean().sort_values("Survived", ascending=False)
corr5


# In[ ]:


# Embarked - Fare Correlation
corr5 = train_df[["Embarked", "Fare"]].groupby(["Embarked"], as_index = False).mean().sort_values("Fare", ascending=False)
corr5


# <a id="7"></a><br>
# ## Outlier Detection

# In[ ]:


def detect_outlier(df, features):
    outlier_indices = []
    
    for c in features:
        #Quartile 1
        Q1 = np.percentile(df[c],25)
        #Quartile 3
        Q3 = np.percentile(df[c],75)
        #IQR
        IQR = Q3 - Q1
        #Outlier step
        outlier_step = IQR * 1.5
        #Detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1-outlier_step) | (df[c] > Q3 + outlier_step)].index
        #Store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


# In[ ]:


train_df.loc[detect_outlier(train_df, ["Age","SibSp","Parch","Fare"])]


# In[ ]:


#Drop outliers
train_df = train_df.drop(detect_outlier(train_df, ["Age","SibSp","Parch","Fare"]), axis =0).reset_index(drop=True)


# <a id="8"></a><br>
# ## Missing Value
# <font color="#9B59B6"/>
# 
# * Find Missing Value
# * Fill Missing Value

# <a id="9"></a><br>
# ## Find Missing Value

# In[ ]:


train_df_len=len(train_df)
train_df = pd.concat([train_df, test_df], axis =0).reset_index(drop=True)


# In[ ]:


train_df.columns[train_df.isnull().any()]


# In[ ]:


train_df.isnull().sum()


# <a id="10"></a><br>
# ## Fill Missing Value
# * Embarked has 2 missing value
# * Fare has 1 missing value

# In[ ]:


#Filling Embarked
train_df[train_df.Embarked.isnull()]


# In[ ]:


train_df.boxplot(column="Fare", by="Embarked")
plt.show()


# In[ ]:


train_df.Embarked = train_df.Embarked.fillna("C")


# In[ ]:


train_df[train_df.Embarked.isnull()]


# In[ ]:


#Filling Fare
train_df[train_df.Fare.isnull()]


# In[ ]:


np.mean(train_df[train_df.Pclass == 3]["Fare"])


# In[ ]:


train_df.Fare = train_df.Fare.fillna(np.mean(train_df[train_df.Pclass == 3]["Fare"]))
train_df[train_df.Fare.isnull()]


# In[ ]:




