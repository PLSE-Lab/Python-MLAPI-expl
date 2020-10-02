#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Lebanon ranked first among the Arab countries in consuming coffee. This short survey focused on the Lebanese programmers only. The aims were to examine if the Lebanese programmers consume coffee above the normal average level comparing to the average consumption in Lebanon which is 1.4 cups of coffee per day.
# 
# Content:    
# 1. [Load and Check](#1)
# 1. [Varriable Description](#2)
#     * [Univarite Variable Analysis](#3)
#         * [Caterogical Variable Analysis](#4)    
#         * [Numerical Variable Analysis](#5)
# 1. [Basic Data Analysis](#6) 
# 1. [Missing Value](#7)
#     * [Find Missing Value](#8)
#     * [Fill Missing Value](#9)
# 1. [Conclusion](#10)    

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization
import matplotlib.pyplot as plt # data visualization

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id = "1"></a>
# # 1) Load and Check

# In[ ]:


df = pd.read_csv("/kaggle/input/coffee-and-code/CoffeeAndCodeLT2018.csv")


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df.describe()


# <a id = "2"></a>
# # 2) Variable Description
# 1. CodingHours: how long has been coding
# 1. CoffeeCupsPerDay: daily amount of coffee
# 1. CoffeeTime: when drinking coffee(before or while coding)
# 1. CodingWithoutCoffee: coding without coffee
# 1. CoffeeType: coffee making type(Turkish, Americano, Nescafe)
# 1. CoffeeSolveBugs: solving mistakes in drink coffee
# 1. Gender: male or female
# 1. Country: this data for Lebanon
# 1. AgeRange: in what age range

# In[ ]:


df.info()


# * int64(2): CodingHours, CoffeeCupsPerDay
# * object(7): CoffeeTime, CodingWithoutCoffee, CoffeeType, CoffeeSolveBugs, Gender, Country, AgeRange 

# <a id = "3"></a>
# # Univarite Variable Analysis
# * Caterogical Variables: CoffeeTime, CodingWithoutCoffee, CoffeeType, CoffeeSolveBugs, Gender, Country, AgeRange
# * Numerical Variables:  CodingHours, CoffeeCupsPerDay

# <a id = "4"></a>
# ## Categorical Variables

# In[ ]:


def count_plot(variable):
    """
        input: variable example: "CoffeTime"
        output: count plot and value count
    """
    # get feature
    var = df[variable]
    
    #visualization
    plt.figure(figsize=(10,4))
    sns.countplot(x=var, palette="dark", order=var.value_counts().index)
    plt.xticks(rotation=45)
    plt.ylabel("Frequency")
    plt.title(variable)
    print("{}".format(var.value_counts()))
    plt.show()


# In[ ]:


categorical = ["CoffeeTime", "CodingWithoutCoffee", "CoffeeType", "CoffeeSolveBugs", "Gender", "Country", "AgeRange"]
for i in categorical:
    count_plot(i)


# <a id="5"></a>
# ## Numerical Variables

# In[ ]:


numerical = ["CodingHours", "CoffeeCupsPerDay"]
for i in numerical:
    count_plot(i)


# <a id="6"></a>
# # 3) Basic Data Analysis
# * Gender-CoffeeType
# * AgeRange-CoffeeCupsPerDay
# * Gender-CoffeeCupsPerDay
# * CodingHours-CoffeeCupsPerDay
# * CoffeeSolveBugs-CoffeeCupsPerDay
# * CodingWithoutCoffee-CoffeeSolveBugs

# ## Gender-CoffeeType

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot("Gender", data=df, hue="CoffeeType", palette="dark")
plt.legend(loc="center")


# ## AgeRange-CoffeeCupsPerDay

# In[ ]:


df[["AgeRange","CoffeeCupsPerDay"]].groupby(["AgeRange"]).mean().sort_values(by="CoffeeCupsPerDay", ascending=False)


# In[ ]:


sns.barplot(x="AgeRange", y="CoffeeCupsPerDay",data=df, palette="dark")


# ## Gender-CoffeeCupsPerDay

# In[ ]:


df[["Gender","CoffeeCupsPerDay"]].groupby(["Gender"]).mean().sort_values(by="CoffeeCupsPerDay", ascending=False)


# In[ ]:


sns.barplot(x="Gender", y="CoffeeCupsPerDay", data=df, palette="dark")


# ## CodingHours-CoffeeCupsPerDay

# In[ ]:


df[["CodingHours","CoffeeCupsPerDay"]].groupby(["CodingHours"]).mean().sort_values(by="CoffeeCupsPerDay", ascending=False)


# In[ ]:


sns.barplot(x="CodingHours", y="CoffeeCupsPerDay", data=df, palette="dark")


# In[ ]:


corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# ## CoffeeSolveBugs-CoffeeCupsPerDay

# In[ ]:


df[["CoffeeSolveBugs","CoffeeCupsPerDay"]].groupby(["CoffeeSolveBugs"]).mean().sort_values(by="CoffeeCupsPerDay", ascending=False)


# In[ ]:


sns.countplot("CoffeeSolveBugs", hue="CoffeeCupsPerDay", palette="dark", data=df)
plt.legend(loc="upper right")


# ## CodingWithoutCoffee-CoffeeSolveBugs

# In[ ]:


sns.countplot("CodingWithoutCoffee", hue="CoffeeSolveBugs",data=df, palette="dark")


# <a id="7"></a>
# # 4) Missing Values
# * Find Missing Value
# * Fill Missing Value

# <a id="8"></a>
# ## Find Missing Value

# In[ ]:


df.columns[df.isnull().any()]


# In[ ]:


df.isnull().sum()


# <a id="9"></a>
# ## Fill Missing Value
# * CoffeeType: 1
# * AgeRange: 2

# In[ ]:


# Missing Value Table
def missing_value_table(df): 
    missing_value = df.isnull().sum()
    missing_value_percent = 100 * df.isnull().sum()/len(df)
    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1)
    missing_value_table_return = missing_value_table.rename(columns = {0 : 'Missing Values', 1 : '% Value'})
    return missing_value_table_return
  
missing_value_table(df)


# In[ ]:


df[df.CoffeeType.isnull()]


# In[ ]:


df[df.AgeRange.isnull()]


# In[ ]:


df["CoffeeType"] = df["CoffeeType"].fillna("Nescafe") 


# I have filled with "Nescafe" because it was most consumed value. So nothing will change
# 

# In[ ]:


df["AgeRange"] = df["AgeRange"].fillna("18 to 29") 


# I filled with "18 to 29".

# In[ ]:


df.isnull().sum()


# <a id="10"></a>
# # 5) Conclusion
# 1. Coffee is most often consumed during coding.(61 people)
# 1. Nescafe is the most popular type of coffee.(32 people)
# 1. the majority of people in the dataset are male.(male:%74, female:%26)
# 1. all the people in the dataset are lebanon.(100 people)
# 1. maximum number of persons between the ages of 18-29(60)
# 1. On average, it consumes the most coffee between the ages of 40-49(3.16 cup of coffee)
# 1. coffee increases linearly as the coding hour increases
# 1. Women consume mostly nescafe, men consume nescafe and americano
