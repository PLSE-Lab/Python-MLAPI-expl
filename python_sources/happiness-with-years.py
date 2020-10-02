#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Read Happiness datas.

# In[ ]:


data2015=pd.read_csv("../input/2015.csv")
data2016=pd.read_csv("../input/2016.csv")
data2017=pd.read_csv("../input/2017.csv")


# In[ ]:


data2015.info()
data2016.info()
data2017.info()


# In[ ]:


data2015.shape


# In[ ]:


data2015.head(10)


# **data2015** have 12 column variable for happiness.
# 1. Country
# 2. Region
# 3. Happiness Rank
# 4. Happiness Score
# 5. Standard Error
# 6. Economy GDP Per Capita
# 7. Family
# 8. Health Life Expectancy
# 9. Freedom
# 10. Trust Goverment Corruption
# 11. Generosity
# 12. Dystopia Residual

# In[ ]:


data2016.shape


# In[ ]:


data2016.head(10)


# **data2016** have 13 column variable for happiness.
# 1. Country
# 2. Region
# 3. Happiness Rank
# 4. Happiness Score
# 5. Lower Confidence Interval
# 6. Upper Confidence Interval
# 7. Economy GDP Per Capita
# 8. Family
# 9. Health Life Expectancy
# 10. Freedom
# 11. Trust Goverment Corruption
# 12. Generosity
# 13. Dystopia Residual

# In[ ]:


data2017.shape


# In[ ]:


data2017.head(10)


# **data2017** have 12 column variable for happiness.
# 1. Country
# 2. Happiness Rank
# 3. Happiness Score
# 4. Whisker High
# 5. Whisker Low
# 6. Economy GDP Per Capita
# 7. Family
# 8. Health Life Expectancy
# 9. Freedom
# 10. Generosity
# 11. Trust Goverment Corruption
# 12. Dystopia Residual

# In[ ]:


f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data2017.corr(),fmt='.1f',annot=True,ax=ax)
plt.title("Heatmap of Correlation 2017")


# In[ ]:


data2017.plot(kind='scatter',figsize=(10,10),x='Economy..GDP.per.Capita.',y='Health..Life.Expectancy.')
plt.title("Economy vs Healt Life Expectancy")


# Economy has big influence on Healt life expectancy.

# In[ ]:


data2017.plot(kind='line',figsize=(10,10),x='Happiness.Rank',y='Generosity')
plt.title("Happiness Rank vs Generosity")


# It is little bit suprising, there is no notable relation between Generosity and happiness. 

# In[ ]:


f, ax = plt.subplots(figsize=(15, 15))
sns.barplot(x=data2015["Happiness Score"][0:10], y=data2015["Country"][0:10],label="2015", color="b")
sns.barplot(x=data2016["Happiness Score"][0:10], y=data2016["Country"][0:10],label="2016", color="r")
sns.barplot(x=data2017["Happiness.Score"][0:10], y=data2017["Country"][0:10],label="2017", color="y")
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 8), ylabel="",
       xlabel="First 10 Country happiness per year")
sns.despine(left=True, bottom=True)


# We can see first 10 happy country.  Only Finland more happy at 2017 according to 2015 and 2016. Other top ten countries less happy.

# In[ ]:


f,ax=plt.subplots(figsize=(30,20))
sns.barplot(x=data2015["Region"],y=data2015["Happiness Score"],ax=ax,color="g",label="2015")
sns.barplot(x=data2016["Region"],y=data2016["Happiness Score"],ax=ax,color="y",label="2016")
ax.legend(ncol=1,loc="upper right",frameon=True)
ax.set(xlabel="Regions vs Happiness Score")


# Most of regions are less happy at 2016 according to 2015.
