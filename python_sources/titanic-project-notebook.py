#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The sinking of the Titanic was a bad accident and happened in 1912,Titanic sank after hitting an iceberg. 1502 people died in total, 2224 people and crew members on board
# 
# <font color='blue'>
# Content:
# 
# 1. [Load and Check Data](#1)
# 2. [Variable Descriptions](#2)
#     * [Univariate Variable Analysis](#3)
#         * [Categorical Variable Analysis](#4)
#         * [Numerical Variable Analysis](#5)
# 3. [Basic Data Analysis](#6)
# 4. [Outlier Detection](#7)
# 5. [Missing Value](#8)
#     * [Find Missing Value](#9)
#     * [Fill Mising Value](#10)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegrid")

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


# <a id="1"></a>
# # Load and Check Data

# In[ ]:


train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

test_PassengerId = test_df['PassengerId']


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# <a id="2"></a>
# # Variable Descriptions
# 
# 1. PassengerId: unique id number to each passenger
# 1. Survived: passenger survive(1) or died(0)
# 1. Pclass: passengers class
# 1. Name: passengers name
# 1. Sex: gender of passengers
# 1. Age: age of pessangers
# 1. SibSp: number of siblings/spouses
# 1. Parch: number of parents/children
# 1. Ticket: ticket number
# 1. Fare: amount of money spent on ticket
# 1. Cabin: cabin category
# 1. Embarked: port where passenger embarked(s,c,q)
# 

# In[ ]:


train_df.info()


# * float64(2): Fare and Age
# * int64(5): Pclass, Sibsp, Parch, PassengerId and Survived
# * object(5): Name, Sex, Ticket, Cabin and Embarked

# <a id="3"></a>
# # Univariate Variable Analysis
# 
# * Categorical Variable: Survived, Sex, PClass, Embarked, Cabin, Name, Ticket, Sibsp and Parch
# * Numerical Variable: Age, PassengerId, Fare

# <a id="4"></a>
# 
# ## Categorical Variable

# In[ ]:


def bar_plot(variable):
    """
    input: variable = ex:'Sex'
    output: bar plot & value count
    """
    # get feature
    var = train_df[variable]
    # count number of categorical variable(value)
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Fre")
    plt.title("Variable")
    plt.show()
    
    print("{}: \n {}".format(variable,varValue))
    


# In[ ]:


category1 = ["Survived","Sex","Pclass","Embarked","Parch","SibSp"]

for i in category1:
    bar_plot(i)


# In[ ]:


category2 = ["Cabin","Name","Ticket"]

for c in category2:
    print("{} \n".format(train_df[c].value_counts()))


# <a id="5"></a>
# 
# ## Numerical Variable

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable],bins=50)
    plt.xlabel(variable)
    plt.ylabel("Fre")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[ ]:


numericVar = ['Fare','Age','PassengerId']

for i in numericVar:
    plot_hist(i)


# <a id="6"></a>
# # Basic Data Analysis
# 
# * Pclass - Survived
# * Sex - Survived
# * Sibsp - Survived
# * Parch - Survived

# In[ ]:


#Plcass vs. Survived

train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


#Sex vs Survived

train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


#Sibsp vs Survived

train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


# People who have 2 or more siblings and spouses have lower survival rates.
train1_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test1_df = pd.read_csv('/kaggle/input/titanic/test.csv')

train1_df['Living_Rate'] = ['Low' if i > 2 else 'High' for i in train1_df['SibSp']]

train1_df.loc[:10,['Living_Rate','SibSp']]


# In[ ]:


#Parch vs Survived

train_df[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by='Survived',ascending=False)


# <a id="7"></a>
# # Outlier Detection

# In[ ]:


def detect_outlier(df,features):
    outlier_indices = []
    
    for c in features:
        # first quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        #iqr
        IQR = Q3 - Q1
        #outlier step
        outlier_step = IQR * 1.5
        #detect outlier and their indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        #store indices
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    return multiple_outliers


# In[ ]:


train_df.loc[detect_outlier(train_df,['Age','SibSp','Parch','Fare'])]


# In[ ]:


#drop outliers

train_df = train_df.drop(detect_outlier(train_df,['Age','SibSp','Parch','Fare']), axis=0).reset_index(drop=True)


# <a id="8"></a>
# # Missing Value
# 
# * Find Missing Value
# * Fill Mising Value

# In[ ]:


train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)


# In[ ]:


train_df.head()


# <a id="9"></a>
# ## Find Missing Value

# In[ ]:


train_df.columns[train_df.isnull().any()] #missing value find


# In[ ]:


train_df.isnull().sum()


# <a id="10"></a>
# ## Fill Missing Value
# 
# * Embarked has 2 missing value
# * Fare has only 1

# In[ ]:


train_df[train_df['Embarked'].isnull()] #find missing value Embarked


# In[ ]:


train_df.boxplot(column='Fare',by = 'Embarked')
plt.show()


# In[ ]:


train_df['Embarked'] = train_df['Embarked'].fillna("C")
train_df[train_df['Embarked'].isnull()] #find missing value Embarked


# In[ ]:


train_df[train_df['Fare'].isnull()] #find missing value Fare


# In[ ]:


train_df[train_df['Pclass'] == 3]['Fare']


# In[ ]:


np.mean(train_df[train_df['Pclass'] == 3]['Fare'])


# In[ ]:


train_df['Fare'] = train_df['Fare'].fillna(np.mean(train_df[train_df['Pclass'] == 3]['Fare']))


# In[ ]:


train_df[train_df['Fare'].isnull()] #find missing value Fare

