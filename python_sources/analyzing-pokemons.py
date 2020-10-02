#!/usr/bin/env python
# coding: utf-8

# In this kernel, I will analyze the Pokemon dataset and find some insights and find some best pokemons based on certain categories. Any ideas regarding finding more insights are encouraged.

# In[77]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[78]:


df = pd.read_csv("../input/Pokemon.csv")


# In[79]:


df.head()


# In[80]:


df.describe()


# I would like to Add Special Attack and Special Defense scores to Attack and Defense respectively

# In[81]:


df["Attack"] = df["Attack"] + df["Sp. Atk"]
df["Defense"] = df["Defense"] + df["Sp. Def"]


# In[82]:


df.head()


# In[83]:


df.drop(columns=["Sp. Atk","Sp. Def",], inplace=True)


# In[84]:


df.head()


# In[85]:


df.isna().sum()


# Here, in this kernel, I would like to analyze only Type 1 category.

# In[86]:


df.drop(columns=["Type 2","#"], inplace=True)


# In[87]:


df.head()


# In[88]:


df["Type 1"].value_counts()


# So, it seems like Pokemons are divided to many Type 1 categories.

# In[89]:


df.Generation.value_counts()


# Pokemons are categorised into 6 Generations

# Let's have a plot based on Generations and Type 1 with Y-axis as scores

# In[90]:


sns.barplot(x="Type 1",y = "Total", data = df).set_title("Type 1 vs Avg-score")
plt.xticks(rotation = 90)


# So, Avg Total score for Pokemons of type Dragon, Steel, Flying are higher compared to that of other types.

# In[91]:


sns.barplot(x="Generation",y = "Total", data = df).set_title("Generation vs Avg-score")


# Avg score of Generation 4 is higher compared to that of other Generations

# In[92]:


sns.barplot(x="Type 1",y = "Speed", data = df).set_title("Type 1 vs Avg-Speed-score")
plt.xticks(rotation = 90)


# As Expected, Speed of Flying Pokemons is higher compared to that of other Pokemons

# In[93]:


sns.barplot(x="Type 1",y = "HP", data = df).set_title("Type 1 vs Avg-HP-score")
plt.xticks(rotation = 90)


# In[94]:


sns.barplot(x="Generation",y = "Speed", data = df).set_title("Generaion vs Avg-Speed-score")


# Generation - 1 Pokemons have higher speed compared to that of other Generations

# In[95]:


sns.barplot(x="Generation",y = "HP", data = df).set_title("Generation vs Avg-HP-score")


# Let's plot overall data based on Total Scores

# In[96]:


plt.figure(figsize=(15,8))
sns.barplot(x = "Type 1", y = "Total",hue = "Generation", data = df )
plt.xticks(rotation = 90)


# In[97]:


df.groupby(["Type 1","Generation"])["Total"].mean()


# In[99]:


print("\n best pokemons based on Total Scores of their respective Generations\n")
ref = dict(df.groupby(["Generation"])["Total"].max())
for i in range(1,7):
    print("\nGeneration : "+str(i)+": "+str(df[(df.Generation == i) & (df.Total == ref[i])].Name))


# In[100]:


print("\n best pokemons based on Total Scores of their respective Types\n")
ref = dict(df.groupby(["Type 1"])["Total"].max())
for key,value in ref.items():
    print("best pokemon based on Type 1 : "+key+" : ",df[(df["Type 1"] == key) & (df.Total == value)].Name)


# Let's see the best and worst pokemons based on total scores

# In[103]:


print("Top 5 Worst Pokemons based on Total scores")
df.sort_values(by = "Total")[:5]


# In[105]:


print("Top 5 Best Pokemons based on Total scores")
df.sort_values(by = "Total",ascending = False)[:5]


# Please suggest me to improve the kernel further.
