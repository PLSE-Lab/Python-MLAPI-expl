#!/usr/bin/env python
# coding: utf-8

# In this kernel,
# Let's analyze the data, find some insights. Let's see which countries have increasing suicide rates and decreasing suicide rates. Which Countries should be careful?

# In[1]:


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


# In[2]:


df = pd.read_csv("../input/master.csv")


# Let's see sample data

# In[3]:


df.sample(5)


# Are there any Nan Values???

# In[4]:


df.isna().sum()


# HDI for Year has Nan values, But I am not considering that column,
# * So let me take columns which I analyze.

# "Suicides/100k pop" is correct for analyzing as it depicts,
# **"how many suicides occured for 100000 population?"**

# In[5]:


first_obs = df[["country","year","sex","age","suicides/100k pop"]]


# In[6]:


first_obs.head()


# **Modifying the "year" attribute**

# In[7]:


print("Min : ",first_obs.year.min())
print("Max : ",first_obs.year.max())


# In[8]:


len(first_obs.year.unique())


# I would like to analyze based on decades, so I would like to modify the year group as below.

# In[9]:


def decade_mapping(data):
    if 1987<= data <= 1996:
        return "1987-1996"
    elif 1997<= data <= 2006:
        return "1997-2006"
    else:
        return "2007-2016"
first_obs.year = first_obs.year.apply(decade_mapping)


# In[10]:


first_obs.sample()


# **Let's start finding insights and see some visualizations**

# 1. Suicides based on Age, Gender

# In[11]:


plt.figure(figsize=(10,5))
sns.barplot(x = "age", y = "suicides/100k pop", hue = "sex",data = first_obs.groupby(["age","sex"]).sum().reset_index()).set_title("Age vs Suicides")
plt.xticks(rotation = 90)


# As you see, as the age increases, suicide rates is increasing irrespective of Gender

# In[12]:


first_obs.groupby(["year","sex"]).sum().reset_index()


# 2. Suicides based on Decades, Gender

# In[13]:


plt.figure(figsize=(10,5))
sns.barplot(x = "year", y = "suicides/100k pop", hue = "sex",data = first_obs.groupby(["year","sex"]).sum().reset_index()).set_title("Decades vs Suicides")


# "1997 - 2006" decade has seen more deaths(suicides)

# In[14]:


sns.barplot(x = "sex", y = "suicides/100k pop", data = first_obs.groupby("sex").sum().reset_index()).set_title("Gender wise Suicides")


# As you see, male suicides are higher than female suicides

# **Country wise Suicide Analysis**

# In[15]:


country_sucides = first_obs.groupby("country").sum().reset_index()
country_sucides.head()


# **Which countries have less suicides?**

# In[16]:


plt.figure(figsize=(10,5))
best_10 = country_sucides.sort_values(by = "suicides/100k pop",ascending= True)[:10]
sns.barplot(x = "country", y = "suicides/100k pop", data = best_10).set_title("countries with less suicides")
plt.xticks(rotation = 90)


# **10 countries with most suicides**

# In[17]:


plt.figure(figsize=(10,5))
best_10 = country_sucides.sort_values(by = "suicides/100k pop",ascending= False)[:10]
sns.barplot(x = "country", y = "suicides/100k pop", data = best_10).set_title("Countries with most suicides")
plt.xticks(rotation = 90)


# **Which countries have most/less suicides recently????**

# In[18]:


recent = first_obs[first_obs.year =="2007-2016"].groupby("country").sum().reset_index()
recent.head()


# In[19]:


plt.figure(figsize=(10,5))
recent_best_10 = recent.sort_values(by = "suicides/100k pop")[:10]
sns.barplot(x = "country", y = "suicides/100k pop", data = recent_best_10).set_title("Countries with less suicides in 2007-2016")
plt.xticks(rotation = 90)


# **Countries with most suicides recently**

# In[20]:


plt.figure(figsize=(10,5))
recent_bad_10 = recent.sort_values(by = "suicides/100k pop",ascending=False)[:10]
sns.barplot(x = "country", y = "suicides/100k pop", data = recent_bad_10).set_title("Countries with most suicides in 2007-2016")
plt.xticks(rotation = 90)


# **DANGER ZONE Nations vs Safe Zone Nations**

# Danger Zone Nations are those nations,
# * where  **suicides** are at **increasing** rate for 3 successive decades, then they are classified to **DANGER ZONE NATIONS**
# 
# Safe Zone Nations are those nations,
# * where **suicides** are at **decreasing** rate for 3 successive decades, then they are classified to **Safe ZONE NATIONS**
# 
# With this analysis,
# * we can understand whether GOVT taking any initiatives for decreasing suicide rate.
# * Why these specfic Nations have increasing Suicide rate?
# 

# In[21]:


zone_assess = first_obs.groupby(["country","year"]).sum().reset_index()
zone_assess.head()


# In[22]:


#countries having data of three decades
three_gen = zone_assess.country.value_counts().reset_index(name = "count")
three_gen.columns = ["country", "counts"]
three_gen_countries = three_gen[three_gen.counts == 3].country.tolist()


# In[23]:


nations = three_gen_countries
years = zone_assess.year.unique()
green_zones = []
danger_zones = []
for country in nations:
    s_year1 = float(zone_assess[(zone_assess.country == country) & (zone_assess.year == "1987-1996")]["suicides/100k pop"])
    s_year2 = float(zone_assess[(zone_assess.country == country) & (zone_assess.year == "1997-2006")]["suicides/100k pop"])
    s_year3 = float(zone_assess[(zone_assess.country == country) & (zone_assess.year == "2007-2016")]["suicides/100k pop"])
    if s_year1 <= s_year2 <= s_year3:
        danger_zones.append(country)
    if s_year1 >= s_year2 >= s_year3:
        green_zones.append(country)
        


# In[24]:


plt.figure(figsize=(18,8))
sns.barplot(x = "country", y = "suicides/100k pop", hue = "year",data = zone_assess[zone_assess.country.isin(green_zones)]).set_title("Decreasing Suicide Rate")
plt.xticks(rotation = 90)


# Suicides are decreasing in above nations.This is a good sign.

# In[25]:


plt.figure(figsize=(18,8))
sns.barplot(x = "country", y = "suicides/100k pop", hue = "year",data = zone_assess[zone_assess.country.isin(danger_zones)]).set_title("Increasing Suicide Rate")
plt.xticks(rotation = 90)


# Suicides are increasing in above nations.Especially, Republic of Korea and Suriname countries have a greater suicide increasing rate.

# I'm a beginner. Give suggesstions for any improvements in kernel.
