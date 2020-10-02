#!/usr/bin/env python
# coding: utf-8

# # Introduction:
# This dataset covers import and export volumes for 5,000 commodities across most countries on Earth over the last 30 years.
# 
# This notebook is prepared for studying Data Analysis. If you feedback me about it i will be grateful.
# 
# <font color = "blue">
# Content:
# 
# 1. [Load and check data](#1)
# 1. [Variable Description](#2)
#     * [Univariate Variable Analysis](#3)
#         * [Categorical Variable Analysis](#4)
#         * [Numerical Variable Analysis](#5)
# 1. [Basic Data Analysis](#6)
# 1. [Outlier Detection](#7)
# 1. [Missing Value](#8)
#    *   [Find Missing Value](#9)
#    *   [Fill Missing Value](#10)

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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id = "1"></a><br>
# ## 1. Load and check data:

# In[ ]:


df = pd.read_csv("../input/global-commodity-trade-statistics/commodity_trade_statistics_data.csv")


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df['category'].unique()


# In[ ]:


Turkey_df = df[df['country_or_area'] == 'Turkey']
Turkey_df


# In[ ]:


Turkey_df.info()


# Analysis of Trade in Turkey

# In[ ]:


Turkey_years = Turkey_df['year'].unique()
Turkey_years.sort()
Turkey_years


# In[ ]:


import_data = []
export_data = []


# In[ ]:


imp_df = Turkey_df[Turkey_df['flow'] == 'Import']
exp_df = Turkey_df[Turkey_df['flow'] == 'Export']


# In[ ]:


for i in Turkey_years:
    x = imp_df[imp_df['year'] == i]
    import_data.append(sum(x.trade_usd)/100000000000)


# In[ ]:


for i in Turkey_years:
    y = exp_df[exp_df['year'] == i]
    export_data.append(sum(y.trade_usd)/100000000000)


# In[ ]:


f,ax = plt.subplots(figsize=(25,15))
sns.barplot(y=import_data,x=Turkey_years,color='green',alpha=1,label='import')
sns.barplot(y=export_data,x=Turkey_years,color='red',alpha=1,label='export')

ax.legend(loc='lower right', frameon = True)
ax.set(xlabel='amount', ylabel='years',title="total amount by years")
plt.xticks(rotation=90)


# In[ ]:


df.describe()


# <a id = "2"></a><br>
# ## 2.Variable Description
# 
# 1. Country_or_area: Country or area name.
# 1. Year: Year of the import/export.
# 1. Comm_code: A unique code of commodity. 
# 1. Commodity:  Name of commodity.
# 1. Flow: Export or import.
# 1. Trade_usd: The cost of commodity.
# 1. Weight_kg: The weight of commodity.
# 1. Quantity_name: The name of quantity. Exp: kg, ton, number.
# 1. Quantity: Quantity.
# 1. Category: The category of commodity.
#      
# 

# In[ ]:


df.info()


# * float64(2): weight_kg, quantity.
# * int64(2): year, trade_usd.
# * objects(6): ocountry_or_area, comm_code, commodity, flow, quantity_name, category.

# <a id = "3"></a><br>
# # 3. Univariate Variable Analysis
# * Categorical Variable Analysis: ocountry_or_area, comm_code, commodity, flow, quantity_name, category.
# * Numerical Variable Analysis: year, trade_usd, weight_kg, quantity.

# <a id = "4"></a><br>
# ## * Categorical Variable Analysis:

# In[ ]:


def bar_plot(variable):
    #___
    #    input: variable ex:"sex"
    #    output:bar plot & value count
    #___
    
    # get a feature
    var = df[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}:\n {}".format(variable,varValue))


# In[ ]:


category1 = ["country_or_area", "flow", "category"]
for c in category1:
    bar_plot(c)


# In[ ]:


category2 = ["comm_code", "commodity", "quantity_name"]
for c in category2:
    print ("{} \n".format(df[c].value_counts()))


# <a id = "5"></a><br>
# ## * Numerical Variable Analysis:

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(df[variable], bins = 150)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[ ]:


numericVar = ["year", "trade_usd", "weight_kg", "quantity"]
for n in numericVar:
    plot_hist(n)


# <a id = "6"></a><br>
# ## 6. Basic Data Analysis
# * weight_kg - trade_usd
# * quantity - trade_usd
# 
# 

# In[ ]:


# weight_kg - trade_usd
df[["weight_kg", "trade_usd"]].groupby(["trade_usd"], as_index = False).mean().sort_values(by = "trade_usd", ascending = False)


# In[ ]:


# quantity - trade_usd
df[["quantity", "trade_usd"]].groupby(["trade_usd"], as_index = False).mean().sort_values(by = "trade_usd", ascending = False)


# <a id = "7"></a><br>
# ## 7. Outlier Detection

# In[ ]:


def detect_outlier(df, features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indices
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


# In[ ]:


df.loc[detect_outlier(df,["year", "trade_usd", "weight_kg", "quantity"])]


# In[ ]:


#drop outliers
df = df.drop(detect_outlier(df,["year", "trade_usd", "weight_kg", "quantity"]), axis = 0).reset_index(drop =True)


# <a id = "8"></a><br>
# # 8. Missing Value
#    *   [Find Missing Value]
#    *   [Fill Missing Value]

# In[ ]:


# df_len = len(df)
# df = pd.concat([df, test_df], axis = 0).reset_index(drop = True)
# df.head()


# <a id = "9"></a><br>
# ## 8.a. Find Missing Value

# In[ ]:


df.columns[df.isnull().any()]


# In[ ]:


df.isnull().sum()


# <a id = "10"></a><br>
# ## 8.b. Fill Missing Value
# 
# * weight_kg has 128475 missing value.
# * quantity has 304857 missing value.

# In[ ]:


df[df["weight_kg"].isnull()]


# In[ ]:


#df.boxplot(column = "trade_usd", by = "weight_kg")
#plt.show()


# In[ ]:


df["weight_kg"] = df["weight_kg"].mean()
df[df["weight_kg"].isnull()]


# In[ ]:


df[df["quantity"].isnull()]

