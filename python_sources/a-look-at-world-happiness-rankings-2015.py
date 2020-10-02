#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# Let's start off by getting the data and having a quick look at its content:

# In[ ]:


data = pd.read_csv("../input/2015.csv")
Df = pd.DataFrame(data)
Df.head(3)


# I will remove the data which I found somewhat irrelevant:

# In[ ]:


df = Df.drop(["Standard Error"], axis = 1)
df.info()


# 10 Happiest and Unhappiest Countries:

# In[ ]:


happy = df["Happiness Rank"] < 11
unhappy = df["Happiness Rank"] > 147
hp = df[np.logical_or(df["Happiness Rank"] < 11, df["Happiness Rank"] > 147)]

hp.plot(kind = "bar", x = "Country", y = "Happiness Score",color = "green")
plt.title("Happiest and Unhappiest Countries")
plt.show()


# Correlation Map:

# In[ ]:


f,ax = plt.subplots(figsize=(13,13))
sns.heatmap(df.corr(), annot = True,fmt="0.1f", ax=ax)
plt.show()


# The correlation map shows that the ranking is based on factors like Economy and Health Care more than factors like Trust at Government and Generosity.
# 
# Economy and Health Care have a correlation of 0.8 which is the highest correlation between the factors. It is worth showing on a scatter plot:

# In[ ]:


df.plot(kind = "scatter", x = "Economy (GDP per Capita)", y = "Health (Life Expectancy)", alpha = 0.5, color = "r", figsize = (10,10))
plt.xlabel("Economy Score")
plt.ylabel("Health Score")
plt.title("Economy-Health")
plt.show()


# We can also see that Generosity and Economy have no connection whatsoever.

# In[ ]:


df.plot(kind = "scatter", x = "Economy (GDP per Capita)", y = "Generosity", alpha = 0.5, color = "b",  figsize = (10,10))
plt.title("Economy-Generosity")
plt.xlabel("Economy Score")
plt.ylabel("Generosity Score")

plt.show()


# In[ ]:




