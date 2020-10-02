#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# 
# The Titanic accident is one of the most dramatic ship break-accident all around the world, forr all times. In 1912, during her voyage, The Titanic sank because of iceberg, killing 1502 out of 2224 passengers and crew.
# <font color = 'Red'>
# # Content:
#     
#     
# 1. [Load and Check data](#1)
# 2. [Variable Description](#2)
#     * [Univariate Variable Analysis](#3)
#         *     [Categorical Variable Analysis](#4)
#         *     [Numerical Variable Analysis](#5)
# 3. [Basic Data Analysis](#6)
# 4. [Outlier Detection](#7)
# 5. [Missing Value](#8)
#     * [Find Missing Value](#9)
#     * [Fill Missing Value](#10)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# <a id = '1'></a><br>
# ## Load and check data

# In[ ]:


train_df = pd.read_csv("../input/titanic/train.csv")
test_df = pd.read_csv("../input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"]


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# <a id = '2' > </a> <br>
# ## Variable Description
# 1. PassengerId: Unique id number to each passenger.
# 2. Survived: Passenger who survived(1) or died(0).
# 3. Pclass: Passenger class.
# 4. Name: Name of passenger.
# 5. Sex: Gender of passenger.
# 6. Age: Age of passenger.
# 7. SibSp: Number of siblings/spouses.
# 8. Parch: Numver of parents/childs.
# 9. Ticket: Ticket number.
# 10. Fare: Cost of ticket.
# 11. Cabin: Place that passengers stay.
# 12. Embarked: Port where passengers embarked(C = Cherbourg, Q = Queenstown, S = Southampton)

# In[ ]:


train_df.info()


# * float64(2) : Age, Fare.
# * int64(5): PassengerId, Survived, Pclass, Sibsp, Parch.
# * object(5): Name, Sex, Ticket, Cabin, Embarked.

# <a id = '3' ></a><br>  
# # Univariate Variable Analysis
# * Categorical Variable Analysis: Survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, Sibsp and Parch.
# * Numerical Variable Analysis: Fare, Age and PassengerId.

# <a id = '4' ></a><br>
# ## Categorical Variable Analysis

# In[ ]:


def bar_plot(variable):
    """
    input: variable, ex: "sex"
    output: bar plot & value count
    
    """
    # get feature 
    var = train_df[variable]
    
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize = (9, 3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title("Variable")
    plt.show()
    print(" {}: \n {} " .format(variable, varValue))


# In[ ]:


category = ['Survived', 'Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']

for i in category:
    bar_plot(i)


# In[ ]:


category2 = ['Name', 'Cabin', 'Ticket']
for i in category2:
    print("{} \n " .format(train_df[i].value_counts()))


# <a id = '5' ></a><br>
# ## Numerical Variable Analysis

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize = (10, 4))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist" .format(variable))
    plt.show()


# In[ ]:


numericVar = [ "Fare", "Age", "PassengerId"]
for i in numericVar:
    plot_hist(i)


# <a id = '6' ></a><br>
# # Basic Data Analysis
# * Pclass - Survived
# * Sex - Survived
# * SibSp - Survived
# * Parch - Survived

# In[ ]:


# Pclass - Survived
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


# Sex - Survived
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


# SibSp - Survived
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


# Sex - Survived
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# <a id = '7' ></a><br>
# # Outlier Detection

# In[ ]:


def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


# In[ ]:


train_df.loc[detect_outliers(train_df, ['Age', 'SibSp', 'Parch', 'Fare'])]


# In[ ]:


train_df = train_df.drop(detect_outliers(train_df, ['Age', 'SibSp', 'Parch', 'Fare']), axis = 0).reset_index(drop = True)


# <a id = '8'></a><br>
# # Missing Value
#    * Find Missing Value
#    * Fill Missing Value

# In[ ]:


train_df_len = len(train_df)
train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)


# In[ ]:


train_df.head()


# <a id = '9'></a><br>
# # Find Missing Value

# In[ ]:


# check if any column has null variable
train_df.columns[train_df.isnull().any()]


# In[ ]:


train_df.isnull().sum()


# <a id = '10'></a><br>
# # Fill Missing Value
# * Embarked has 2 missing value
# * Fare has only 1

# In[ ]:


## We can delete the missing values but we do not want to lose any value, so that they will be filled.

train_df[train_df['Embarked'].isnull()]


# In[ ]:


train_df.boxplot(column = 'Fare', by = 'Embarked')
plt.show()


# In[ ]:


train_df['Embarked'] = train_df['Embarked'].fillna('C')
# CHECK
train_df[train_df['Embarked'].isnull()]


# In[ ]:


train_df[train_df['Fare'].isnull()]


# In[ ]:


train_df.groupby('Pclass').Fare.mean()


# In[ ]:


# train_df.groupby('Embarked').Fare.mean()
# train_df['Fare'] = train_df['Fare'].fillna(12.741220)
# train_df[train_df.PassengerId == 1044]

## Or

train_df['Fare'] = train_df['Fare'].fillna(np.mean(train_df[train_df['Pclass'] == 3]['Fare']))
train_df[train_df.PassengerId == 1044]


# In[ ]:


train_df[train_df['Fare'].isnull()]

