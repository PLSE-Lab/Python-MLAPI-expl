#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# <img src="https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2F3.bp.blogspot.com%2F-VM1bQwyqdh8%2FUd6BJq6ER7I%2FAAAAAAAAMN4%2FPQW_bKZ6RXQ%2Fs1600%2FTitanicSmithsonian34460_copy_crop.jpg&f=1&nofb=1" width="400"><br>
# 
# The RMS Titanic was a luxury passenger liner that, when it set sail on her maiden voyage from Southampton, England to New York City on 10 April 1912, has been built using the most advanced technology at the time.
# 
# Four days after setting sail, on April 15, the Titanic struck an iceberg. The ship sank and 1,517 of the 2,223 people on board died. The high casualty rate was due in part to the fact that the ship carried lifeboats for only 1,178 people. The "women and children first" protocol that was enforced by the ship's crew meant many more men died, and more people from the crew and the third class passengers died compared to those in first class.
# 
# <font color = 'blue'>
# Content:
# 
# 1. [Load and Check Data](#1)
# 1. [Variable Description](#2)
#     * [Univeriate Variable Analysis](#3)
#         * [Categorical Variables](#4)
#         * [Numerical Variables](#5)
# 1. [Data Analysis with Visualization](#6)
#     * [Correlation Between Sibsp, Parch, Age, Fare and Survived](#60)
#     * [Pclass vs Survived](#61)
#     * [Sex vs Survived](#62)
#     * [SibSp vs Survived](#63)
#     * [Parch vs Survived](#64)
#     * [Age vs Survived](#65)
#     * [Correlation Between Pclass, Survived and Age](#66)
#     * [Correlation Between Embarked, Sex, Pclass and Survived](#67)
#     * [Correlation Between Embarked, Sex, Fare and Survived](#68)
# 1. [Outlier Detection](#7) 
# 1. [Missing Values](#8)
#     * [Finding Missing Values](#9)
#     * [Filling Missing Values](#10)   
#     * [Filling Missing Age Feature](#11)
# 1. [To Be Continued...](#12)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

import seaborn as sns
import plotly.graph_objs as go

from collections import Counter

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <a id = '1'></a>
# # Load and Check Data

# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerID = test_df['PassengerId']


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# <a id = '2'></a>
# # Variable Description
# 1. PassengerId : Unique id number to each passenger
# 2. Survived    : Passenger survive(1) or died(0)
# 3. Pclass      : Ticked Class (1 = 1st, 2 = 2nd, 3 = 3rd)
# 4. Name        : Name
# 5. Sex         : Gender of Passenger
# 6. Age         : Age in Years
# 7. SibSp       : Number of siblings/spouses aboard the Titanic
# 8. Parch       : Number of parents/children aboard the Titanic
# 9. Ticket      : Ticket Number
# 10. Fare       : Passenger Fare
# 11. Cabin      : Cabin Number
# 12. Embarked   : Port of Embarkation (C = Cherboug, Q = Queenstown, S = Southampton)

# In[ ]:


train_df.info()


# * float64(2) : Fare and Age
# * int64(5) : Pclass, SibSp, Parch, Survived, PassengerId
# * object(5) : Cabin,Name, Sex, Embarked, Ticket

# <a id = '3'></a>
# # Univeriate Variable Analysis
# * Categorical Variable : Survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, SibSp and Parch
# * Numerical Variable : Age, PassengerId and Fair

# <a id = '4'></a>
# ## Categorical Variables

# In[ ]:


def bar_plot(variable):
    """
        input : variable ex:"Sex"
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
    print("{}: \n {}".format(variable,varValue))


# In[ ]:


category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]
for c in category1:
    bar_plot(c)


# In[ ]:


category2 = ["Cabin", "Name", "Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))


# <a id = '5'></a>
# ## Numerical Variables

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[ ]:


numericVar = ["Fare", "Age", "PassengerId"]
for n in numericVar:
    plot_hist(n)


# <a id = '6'></a>
# # Data Analysis with Visualization
# * Correlation Between Sibsp, Parch, Age, Fare and Survived
# * Pclass vs Survived
# * Sex vs Survived
# * SibSp vs Survived
# * Parch vs Survived
# * Age vs Survived

# <a id = "60"></a><br>
# ### Correlation Between Sibsp, Parch, Age, Fare and Survived

# In[ ]:


list1 = ['SibSp','Parch','Age','Fare','Survived']
sns.heatmap(train_df[list1].corr(), annot = True, fmt = '.2f', linewidths=.5)


# Fare feature seems to be related to the Survived feature (at the rate of 0.26)

# <a id = '61'></a><br>
# ### Pclass vs Survived

# In[ ]:


g = sns.factorplot(x = 'Pclass', y = 'Survived', data = train_df, kind = 'bar', size = 6)
g.set_ylabels('Survived Probability')
plt.show()
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending = False)


# <a id = '62'></a><br>
# ### Sex vs Survived

# In[ ]:


g = sns.factorplot(x = 'Sex', y = 'Survived', data = train_df, kind = 'bar', size = 6)
g.set_ylabels('Survived Probability')
plt.show()

train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending = False)


# * The "women and children first" protocol that was enforced by the ship's crew meant many more men died, and more people from the crew and the third class passengers died compared to those in first class.

# <a id = '63'></a><br>
# ### SibSp vs Survived

# In[ ]:


g = sns.factorplot(x = 'SibSp', y = 'Survived', data = train_df, kind = 'bar', size = 6)
g.set_ylabels('Survived Probability')
plt.show()

train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending = False)


# * Having a lot of SibSp have less change to survive.
# * Passengers with SibSp == 0, 1, or 2 have more chances to survive.

# <a id = '64'></a><br>
# ### Parch vs Survived

# In[ ]:


g = sns.factorplot(x = 'Parch', y = 'Survived', data = train_df, kind = 'bar', size = 6)
g.set_ylabels('Survived Probability')
plt.show()

train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending = False)


# * SibSp and Parch can be used for new feature extraction with th = 3
# * Small families have more change to survive.
# * There is a std in survival of passenger with Parch = 3
# * The black bar represents the standard deviation.

# <a id = '65'></a><br>
# ### Age vs Survived
# 

# In[ ]:


g = sns.FacetGrid(train_df, col = 'Survived', size = 6)
g.map(sns.distplot, 'Age', bins = 25)
plt.show()


# * Those aged 10 and under 10 have high survival rates.
# * Oldest passengers (80) survived.
# * Large number of 20 years old did not survive.
# * Most passenger are in 18-37 age range.
# * Age distribution will be used for missing value of Age.
# * Age feature will be used in training.

# <a id = '66'></a><br>
# ## Correlation Between Pclass, Survived and Age

# In[ ]:


g = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass', size = 3)
g.map(plt.hist, 'Age', bins = 25)
g.add_legend()
plt.show()


# * Pclass seems to be an important feature for model training

# <a id = '67'></a><br>
# ## Correlation Between Embarked, Sex, Pclass and Survived

# In[ ]:


g = sns.FacetGrid(train_df, row = 'Embarked', size = 4)
g.map(sns.pointplot, 'Pclass','Survived','Sex')
g.add_legend()
plt.show()


# * Female passengers have much better survival rate than Male passengers.
# * Male passengers have better survival rate in Pclass 3 in C.
# * Embarked and Sex will be used in training.

# <a id = '68'></a><br>
# ## Correlation Between Embarked, Sex, Fare and Survived

# In[ ]:


g = sns.FacetGrid(train_df, row = 'Embarked', col = 'Survived', size = 3)
g.map(sns.barplot, 'Sex','Fare')
g.add_legend()
plt.show()


# * Passengers who pay higher Fare have better survival rate.

# <a id = "7"></a><br>
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
        # Outlier Step
        outlier_step = IQR * 1.5
        # Detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # Store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    return multiple_outliers


# In[ ]:


train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]


# In[ ]:


# Drop outliers
train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)


# <a id = "8"></a><br>
# # Missing Values
# * Finding Missing Values.
# * Filling Missing Values.
# * Filling Missing Age Values.

# In[ ]:


train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop = True)


# <a id = "9"></a><br>
# ## Finding Missing Values

# In[ ]:


train_df.columns[train_df.isnull().any()]


# In[ ]:


train_df.isnull().sum()


# <a id = "10"></a><br>
# ## Filling Missing Values
# * Embarked has 2 missing values.
# * Fare has only 1 missing value.
# * Age has 256 missing values.

# In[ ]:


train_df[train_df['Embarked'].isnull()]


# In[ ]:


train_df.boxplot(column='Fare', by = 'Embarked')
plt.show()


# In[ ]:


train_df['Embarked'] = train_df['Embarked'].fillna('C')
train_df[train_df['Embarked'].isnull()]


# * Missing Embarked values are filled.

# In[ ]:


train_df[train_df['Fare'].isnull()]


# In[ ]:


np.mean(train_df[train_df['Pclass']==3]['Fare'])


# In[ ]:


train_df['Fare'] = train_df['Fare'].fillna(np.mean(train_df[train_df['Pclass']==3]['Fare']))
train_df[train_df['Fare'].isnull()]


# * Missing Fare value has been filled.

# <a id = '11'></a><br>
# ### Filling Missing Age Feature

# In[ ]:


train_df[train_df['Age'].isnull()]


# In[ ]:


sns.factorplot(x= 'Sex', y = 'Age', data = train_df, kind = 'box')
plt.show()


# * Sex is not informative for age prediction, age distribution seems be same.

# In[ ]:


sns.factorplot(x= 'Sex', y = 'Age', hue = 'Pclass', data = train_df, kind = 'box')
plt.show()


# * 1st class passengers are older than 2nd class passengers and 2nd class passengers are older than 3rd class passengers.

# In[ ]:


sns.factorplot(x= 'Parch', y = 'Age', data = train_df, kind = 'box')
sns.factorplot(x= 'SibSp', y = 'Age', data = train_df, kind = 'box')
plt.show()


# In[ ]:


# In order to see Age feature in heatmap, we need to convert it to numerical value.
train_df['Sex'] = [1 if i == 'male' else 0 for i in train_df['Sex']]


# In[ ]:


sns.heatmap(train_df[['Age','Sex','SibSp','Parch','Pclass']].corr(), annot = True,linewidths=.5)
plt.show()


# * Age is not correlated with Sex but it is correlated with Parch, SibSp and Pclass.

# In[ ]:


index_nan_age = list(train_df['Age'][train_df['Age'].isnull()].index)
for each in index_nan_age:
    age_prediction = train_df['Age'][((train_df['SibSp'] == train_df.iloc[each]['SibSp']) &(train_df['Parch'] == train_df.iloc[each]['Parch'])& (train_df['Pclass'] == train_df.iloc[each]['Pclass']))].median()
    age_median = train_df['Age'].median()
    if not np.isnan(age_prediction):
        train_df['Age'].iloc[each] = age_prediction
    else:
        train_df['Age'].iloc[each] = age_median


# In[ ]:


train_df[train_df['Age'].isnull()]


# * Missing Age values are filled

# <a id = '12'></a><br>
# # To Be Continued...
# > Next Step : Machine Learning...

# > If you liked this, upvote!
