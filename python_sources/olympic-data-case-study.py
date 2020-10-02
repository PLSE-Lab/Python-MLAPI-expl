#!/usr/bin/env python
# coding: utf-8

# # Find following details from the given Olympic_DataSets
# - Describe the data
# - Get info about all the columns
# - Find all the column name which has null (NaN) values
# - Merge regions and Olymic dataset and create new DataFrame
# - Plot the age distribution of Gold medalist
# - From above plot, find out the sport type (which discipline) athelets belong to who won Gold medal and their age is >=50
# - How many such athelets are present in above data
# - Try above for Men and Womens separately
# - Plot Medals per country (plot only top 10 countries)
# - Find out Disciplines(sport type) with the greatest number of Gold Medals
# - What is the median height/weight of an Olympic medalist? (While plotting think of NaN and decide whether to replace or    
#   drop) Use scatter plot.
# - Plot trend line for Evolution of the Olympics over time for Male and Female only for "Winter" games
# - Variation of male/female athletes over time (Summer Games)
# - Variation of age for MEN along time using sns plot
# - Variation of age for WOMEN along time using sns plot

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


Ol_data = pd.read_csv("../input/athlete_events.csv")
reg_data = pd.read_csv("../input/noc_regions.csv")


# ## Describe the data

# In[ ]:


Ol_data.head(1)


# In[ ]:


reg_data.head(1)


# In[ ]:


Ol_data.describe()


# In[ ]:


reg_data.describe()


# ## Get info about all the columns

# In[ ]:


Ol_data.info()


# In[ ]:


reg_data.info()


# #### Find all the column name which has null (NaN) values

# In[ ]:


for k in Ol_data.keys():
    if Ol_data[k].isnull().any():
        print(k)


# #### Merge regions and Olymic dataset and create new DataFrame

# In[ ]:


merge_data = pd.merge(Ol_data,reg_data,how="left",on="NOC",indicator=True)
merge_data.head(2)


# #### Plot the age distribution of Gold medalist

# In[ ]:


Ol_data.head(1)


# In[ ]:


plt.figure(figsize=(16,4))
Ol_data[Ol_data.Medal == "Gold"].Age.plot(kind="hist",bins=50,width=.8,color="g")
plt.xticks(range(1,100,2))
plt.show()


# #### From above plot, find out the sport type (which discipline) athelets belong to who won Gold medal and their age is >=50

# In[ ]:


plt.figure(figsize=(16,4))
Ol_data[(Ol_data.Medal == "Gold") & (Ol_data.Age >= 50)].Sport.value_counts().plot(kind="bar",color="c")
# plt.xticks(range(1,100,2))
plt.show()


# #### How many such athelets are present in above data

# In[ ]:


Ol_data[(Ol_data.Medal == "Gold") & (Ol_data.Age >= 50)].index.size


# #### Try above for Men and Womens separately

# In[ ]:


plt.figure(figsize=(16,4))
Ol_data[(Ol_data.Medal == "Gold") & (Ol_data.Sex == "M")].Sport.value_counts().plot(kind="bar",color="c")
# Ol_data[(Ol_data.Medal == "Gold") & (Ol_data.Sex == "M")].Sport.value_counts().plot(kind="line",color="k")
plt.show()


# In[ ]:


plt.figure(figsize=(16,4))
Ol_data[(Ol_data.Medal == "Gold") & (Ol_data.Sex == "F")].Sport.value_counts().plot(kind="bar",color="r")
Ol_data[(Ol_data.Medal == "Gold") & (Ol_data.Sex == "F")].Sport.value_counts().plot(kind="line",color="g")
plt.xticks(rotation=75)
plt.show()


# #### Plot Medals per country (plot only top 10 countries)

# In[ ]:


color = ["r","g","b","c","r","g","b","c","m","k","m","k"]
total_medals = Ol_data[Ol_data.Medal == "Gold"].groupby("NOC")["Medal"].value_counts().reset_index(name="CNT")
# print(total_medals)
# total_medals.sort_values(by="count",ascending=False).head(10).plot(kind="bar")
tmp = total_medals.sort_values(by="CNT",ascending=False).head(10)
# print(tmp.NOC)
plt.bar(tmp.NOC,tmp.CNT,color=color)
plt.xticks(rotation=75)
plt.show()


# ####  Find out Disciplines(sport type) with the greatest number of Gold Medals

# # Try at home!!!

# #### What is the median height/weight of an Olympic medalist? (While plotting think of NaN and decide whether to replace or drop) Use scatter plot.

# In[ ]:


# Ol_data.Weight.isnull().any()
# Ol_data.Height.isnull().any()
tmp_df = Ol_data[(Ol_data.Height.notnull()) & (Ol_data.Weight.notnull())]
tmp_df.head(2)
plt.figure(figsize=(16,4))
plt.scatter(tmp_df.Height,tmp_df.Weight,color="g",alpha=.4)
plt.hlines(125,130,220,color="r")
plt.vlines(180,25,200,color="r")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Height Vs Weight")
plt.show()


# #### Plot trend line for Evolution of the Olympics over time for Male and Female only for "Winter" games

# # Try at home!!!

# #### Variation of male/female athletes over time (Summer Games)

# # Try at home!!!

# #### Variation of age for MEN along time using sns plot

# In[ ]:


male_data = Ol_data[Ol_data.Sex=="M"]
plt.figure(figsize=(16,8))
sns.boxplot("Year","Age",data=male_data)
plt.show()


# In[ ]:


male_data = Ol_data[Ol_data.Sex=="M"]
plt.figure(figsize=(16,8))
sns.pointplot("Year","Age",data=male_data)
plt.show()


# #### Variation of age for WOMEN along time using sns plot

# In[ ]:


female_data = Ol_data[Ol_data.Sex=="F"]
plt.figure(figsize=(16,8))
sns.boxplot("Year","Age",data=female_data)
plt.show()


# In[ ]:


male_data = Ol_data[Ol_data.Sex=="M"]
female_data = Ol_data[Ol_data.Sex=="F"]
plt.figure(figsize=(16,8))
sns.pointplot("Year","Age",data=male_data,color="r")
sns.pointplot("Year","Age",data=female_data)
plt.show()

