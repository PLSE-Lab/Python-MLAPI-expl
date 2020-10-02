#!/usr/bin/env python
# coding: utf-8

# <font color = 'blue'>
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
# 1. [Visualization](#9)       

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import matplotlib.pyplot as plt
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


# <a id = "1"></a><br>
# # Load and Check Data

# In[ ]:


df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.head(10)


# In[ ]:


df.describe()


# <a id = "2"></a><br>
# # Variable Description
# 1. Rank: Ranking of overall sales
# 1. Name: The games name
# 1. Platform:  Platform of the games release (i.e. PC,PS4, etc.)
# 1. Year: Year of the game's release
# 1. Genre: Genre of the game  
# 1. Publisher: Publisher of the game
# 1. NA_Sales: Sales in North America (in millions)	
# 1. EU_Sales: Sales in Europe (in millions)
# 1. JP_Sales: Sales in Japan (in millions)
# 1. Other_Sales: Sales in the rest of the world (in millions)	
# 1. Global_Sales: Total worldwide sales.
# 

# In[ ]:


df.info()


# * float64(6): Year, Na_Sales, EU_Sales, JP_Sales, Other_Sales and Global_Sales
# * int64(1): Rank
# * object(4): Name, Platform, Genre and Publisher

# <a id = "3"></a><br>
# # Univariate Variable Analysis
# * Categorical Variable: Platform, Genre, Name, Publisher and Year
# * Numerical Variable: Rank, Na_Sales, EU_Sales, JP_Sales, Other_Sales and Global_Sales

# <a id = "4"></a><br>
# ## Categorical Variable

# In[ ]:


def bar_plot(variable):
    varCount = df[variable].value_counts()
    
    plt.figure(figsize = (10,10))
    plt.bar(varCount.index, varCount)
    plt.xticks(varCount.index, varCount.index.values,rotation=90, horizontalalignment='right',fontweight='light',fontsize='large')
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()


# In[ ]:


category1 = [ "Platform", "Genre","Year"] # I didn't do plt bar other categorical variables like name and publisher for it not show correct visual
for c in category1:
    bar_plot(c)


# In[ ]:


category2 = ["Name","Publisher"]
for c in category2:
    print(df[c].value_counts())


# <a id = "5"></a><br>
# ## Numeric Variable

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize = (5,5))
    plt.hist(df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.show()


# In[ ]:


numericVar = [ "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
for n in numericVar:
       plot_hist(n)


# <a id = "6"></a><br>
# # Basic Data Analysis
# * Genre - Global_Sales
# * Platform - Global_Sales
# * Year - Global_Sales
# * Publisher - Global_Sales

# In[ ]:


# Genre vs Global_Sales 
df[["Genre","Global_Sales"]].groupby(["Genre"], as_index = False).mean().sort_values(by="Global_Sales",ascending = False)


# In[ ]:


# Platform vs Global_Sales
df[["Platform","Global_Sales"]].groupby(["Platform"], as_index = False).mean().sort_values(by="Global_Sales",ascending = False)


# In[ ]:


# Year vs Global_Sales
df[["Year","Global_Sales"]].groupby(["Year"], as_index = False).mean().sort_values(by="Global_Sales",ascending = False)


# In[ ]:


# Publisher vs Global_Sales
p = df[["Publisher","Global_Sales"]].groupby(["Publisher"], as_index = False).mean().sort_values(by="Global_Sales",ascending = False)
p.head(15)   
    


# <a id = "7"></a><br>
# # Outlier Detection

# In[ ]:


df["Publisher"].unique()


# In[ ]:


# outlier detection(IQR-Quantile) get reference from publisher 

for column in df.columns[6:]: # selected only sales columns
    for p in df["Publisher"].unique():
        selected_p = df[df["Publisher"] == p]
        selected_column = selected_p[column]
        
        q1 = selected_column.quantile(0.25)
        q3 = selected_column.quantile(0.75)
        
        iqr = q3 - q1
        
        minimum = q1 - (1.5 * iqr)
        maximum = q3 + (1.5 * iqr)
        
        print(column,p,"| min=",minimum,"max=",maximum)
        
        
        max_idxs = df[(df["Publisher"] == p) & (df[column] > maximum)].index  
        print(max_idxs)
        min_idxs = df[(df["Publisher"] == p) & (df[column] < minimum)].index  
        print(min_idxs)
        
        df.drop(index= max_idxs,inplace= True)
        df.drop(index= min_idxs,inplace= True)
        


# In[ ]:


df.info()


# <a id = "8"></a><br>
# # Missing Values

# In[ ]:


df.isna().sum() # find missing values


# In[ ]:


# filling missing values with mode
df["Year"].fillna(df["Year"].mode()[0], inplace=True)
df["Publisher"].fillna(df["Publisher"].mode()[0], inplace=True)

df.isna().sum()


# <a id = "9"></a><br>
# # Visualization

# In[ ]:


plt.figure(figsize = (8,8))
ax = sns.heatmap(df.corr(), linewidths=.5)


# In[ ]:


for column in df.columns[6:]:
    sns.relplot(x="Year", y=column, kind="line",data=df)
    plt.show()


# In[ ]:


genre_val = df.Genre.value_counts().values
labels = df.Genre.value_counts().index
plt.figure(figsize=(10,10))
plt.pie(genre_val, labels=labels)

plt.show()

