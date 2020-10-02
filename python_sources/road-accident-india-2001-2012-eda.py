#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap # For geographical map
sns.set_style("darkgrid")


# In[ ]:


df = pd.read_excel("../input/datafile.xls")


# In[ ]:


df.head(1)


# # Findout the geographical location where exactly accident happened by just looking at Longitude and Lattitude in dataset
# So it's in INDIA as per map!!!

# In[ ]:


map = Basemap(lat_0=19.0, lon_0=71)
plt.figure(figsize=[20,10])
map.drawcoastlines(linewidth=.5,color="g")
map.drawcountries(linewidth=.5)
# map.drawcounties()
# map.drawstates(color="r")
map.fillcontinents(color="green",alpha=.1)
plt.scatter(df.Long,df.Lat,alpha=.1,color="m")
plt.show()


# # Lets have a look at overall summary using pair plot

# In[ ]:


sns.pairplot(df[["Year","Male","Female","Total"]],diag_kind="kde")
plt.show()


# # Lets have a look at cause of death for Male
# Most of the death happened by 2-wheeler and then Truck

# In[ ]:


plt.figure(figsize=(16,4))
sns.scatterplot(df.CAUSE,df.Male)
plt.xticks(rotation=90)
plt.show()


# ### Using box plot for detecting outliers

# In[ ]:


plt.figure(figsize=(16,4))
sns.boxplot(df.CAUSE,df.Male)
# sns.swarmplot(df.CAUSE,df.Male)
plt.xticks(rotation=90)
plt.show()


# # Lets have a look at cause of death for Female
# Most of the death happened by Truck and then 2-wheeler

# In[ ]:


plt.figure(figsize=(16,4))
sns.scatterplot(df.CAUSE,df.Female)
plt.xticks(rotation=90)
plt.show()


# ### Using box plot for detecting outliers

# In[ ]:


plt.figure(figsize=(16,4))
sns.boxplot(df.CAUSE,df.Female)
plt.xticks(rotation=90)
plt.show()


# # Total death toll for each year

# In[ ]:


plt.figure(figsize=(16,4))
grp1 = df.groupby("Year")
for k in grp1.groups:
    tmp = df[df.Year == k]
    overall = tmp.Total.sum()
    male_cnt = tmp.Male.sum()
    female_cnt = tmp.Female.sum()
    
    plt.bar(k,overall)
    plt.text(k,overall,overall)
    
plt.xticks(range(2001,2013))
plt.show()


# # Total death toll for each year for Male/Female

# In[ ]:


plt.figure(figsize=(16,4))
grp1 = df.groupby("Year")
for k in grp1.groups:
    tmp = df[df.Year == k]
    male_cnt = tmp.Male.sum()
    female_cnt = tmp.Female.sum()
    
    plt.bar(k,male_cnt,color="r")
    plt.text(k,male_cnt,male_cnt)
    
    plt.bar(k,female_cnt,color="m")
    plt.text(k,female_cnt,female_cnt)

plt.xticks(range(2001,2013))
plt.legend(["Male","Female"])
plt.show()


# # Total death toll for each year for Male/Female
# using sns plot

# In[ ]:


plt.figure(figsize=(16,4))
# grp1 = df.groupby("Year")
# grp1.Year.value_counts()
male_cnt = df.pivot(columns="Year").Male.sum().reset_index(name="cnt")
female_cnt = df.pivot(columns="Year").Female.sum().reset_index(name="cnt")
sns.barplot(male_cnt.Year,male_cnt.cnt,palette="coolwarm",label="Male")
sns.barplot(female_cnt.Year,female_cnt.cnt,label="Female")
plt.legend()
plt.show()


# # State wise Total death toll
# using sns plot

# In[ ]:


plt.figure(figsize=(16,4))
Total = df.pivot(columns="States").Total.sum().reset_index(name="cnt")
Total.sort_values(by="cnt",ascending=False,inplace=True)
sns.barplot(Total.States,Total.cnt,palette="winter")
plt.xticks(rotation=90)
plt.show()


# # State wise Total death toll for Male/Female
# using sns plot

# In[ ]:


plt.figure(figsize=(16,4))
male_cnt = df.pivot(columns="States").Male.sum().reset_index(name="cnt")
female_cnt = df.pivot(columns="States").Female.sum().reset_index(name="cnt")
male_cnt.sort_values(by="cnt",ascending=False,inplace=True)
sns.barplot(male_cnt.States,male_cnt.cnt,palette="winter",label="Male")
sns.barplot(female_cnt.States,female_cnt.cnt,color="r",label="Female")
plt.xticks(rotation=90)
plt.legend()
plt.show()


# #  Various accident type and total death toll
# using sns plot

# ###### By looking at the graph, it is clear that Truck and two wheeler are the most common cause of death

# In[ ]:


plt.figure(figsize=(16,4))
total_deatth_by_cause = df.pivot(columns="CAUSE").Total.sum().reset_index(name="cnt")
total_deatth_by_cause
total_deatth_by_cause.sort_values(by="cnt",ascending=False,inplace=True)
sns.barplot(total_deatth_by_cause.CAUSE,total_deatth_by_cause.cnt,palette="nipy_spectral_r",)
plt.xticks(rotation=90)
plt.show()


# # Heat map of death toll cause Vs/Year

# In[ ]:


plt.figure(figsize=(16,8))
pi = df.pivot_table("Total","CAUSE","Year")
sns.heatmap(pi,square=False,annot=True,fmt="1.2g")
plt.show()


# # Heat map of Male death toll States Vs/Year

# In[ ]:


plt.figure(figsize=(16,8))
pi = df.pivot_table("Male","States","Year")
sns.heatmap(pi,square=False,annot=True,fmt="1.2g")
plt.show()


# # Heat map of Female death toll States Vs/Year

# In[ ]:


plt.figure(figsize=(16,8))
pi = df.pivot_table("Female","States","Year")
sns.heatmap(pi,square=False,annot=True,fmt="1.2g")
plt.show()


# In[ ]:


df.head(1)

