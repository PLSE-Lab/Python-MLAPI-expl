#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


print("Reading 2015.csv file for analysis\n")

y_2015 = pd.read_csv("../input/2015.csv")

print("First 5 rows of data :\n",y_2015.head())

print("Last 5 rows of data :\n",y_2015.tail())


# In[3]:


print("Data Info :\n")
print(y_2015.info())


# In[41]:


print("Data Describe\n",(y_2015['Freedom'].describe()))


# In[5]:


print("Check if any column has Null Value:\n")
print(y_2015.isnull().sum())


# In[6]:


print("Count of unique values in each columns:\n")
print(y_2015.nunique())


# In[7]:


print("Unique Region names:\n")
for region in y_2015['Region'].unique():
    print(region)


# In[8]:


print("Count of each Regions:\n")
print(y_2015['Region'].value_counts())


# In[9]:


print("Unique Country names:\n")
for index,country in enumerate(y_2015['Country'].unique()):
    print(index+1,country)


# In[10]:


print ("Arranging the Countries according to their Regions:\n")
data = {}

for region,country in zip(y_2015['Region'],y_2015['Country']):
    if region not in data:
        data[region] = country
    else:
        data[region] += ',' + country

for keys,values in data.items():
    print(keys,":\n",values,'\n')


# In[11]:


print("Data types of all the columns:")
print(y_2015.dtypes)


# **Data Visualization**

# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
region = y_2015['Region'].value_counts()

sns.scatterplot(region.index,region.values)
plt.title("Scatter Plot of Number of Countries in each Region")
plt.xticks(rotation=90)
plt.xlabel("Regions")
plt.ylabel("Values")
plt.show()

sns.barplot(region.index,region.values)
plt.title("Bar Plot of Number of Countries in each Region")
plt.xticks(rotation=90)
plt.xlabel("Regions")
plt.ylabel("Values")
plt.show()

plt.pie(region.values,labels = region.index)
plt.title("Pie Chart of Number of Countries in each Region")

plt.show()


# In[13]:


#Regions that are more happier than the rest
data = y_2015.groupby('Region')['Happiness Score'].mean()

plt.barh(data.index,data.values)
plt.title("Average Happiness Score of each Region")
plt.xlabel("Happiness Scores")
plt.ylabel("Regions")
plt.show()


# In[39]:


#pie chart of economy percent of all regions

region_names = y_2015['Region'].value_counts().index
region_values =  y_2015['Region'].value_counts().values

economy_rate = []
for i,reg in enumerate(region_names):    
    gdp = sum(y_2015[y_2015['Region']==reg]['Economy (GDP per Capita)'])/region_values[i]
    economy_rate.append(gdp)

plt.figure(figsize=(10,10))
plt.pie(economy_rate,labels = region_names,autopct='%1.1f%%')
plt.title("Pie Chart of Economy Percent in each Region",color='b',fontsize=20)

plt.show()


# In[45]:


region_names = y_2015['Region'].value_counts().index
region_values =  y_2015['Region'].value_counts().values

freedom_rate = []
for i,reg in enumerate(region_names):
    freedom = sum(y_2015[y_2015['Region'] == reg]['Freedom'])/region_values[i]
    freedom_rate.append(freedom)
    
plt.figure(figsize=(10,10))
sns.barplot(region_names,freedom_rate)
plt.title("Bar Plot of Freedom in each Region",color='r',fontsize=20)
plt.xlabel("Regions")
plt.ylabel("Freedom")
plt.xticks(rotation=90)
plt.show()


# #Regions Vs Economy
# #Region Vs Freedom
# #Region Vs Happiness Rank
# #Region Vs Trust
# 
# #Freedom Vs Trust
# #Freedom Vs Happiness Rank
# #Freedom Vs Economy
# 
# #Happiness Rank and Happiness Score

# In[56]:


#Most generous region
region_names = y_2015['Region'].value_counts().index
region_values = y_2015['Region'].value_counts().values

generosity = []

for i,reg in enumerate(region_names):
    generosity.append(sum(y_2015[y_2015['Region']==reg]['Generosity'])/region_values[i])
                
plt.figure(figsize=(10,10))
sns.barplot(region_names,generosity)
plt.title("Bar Plot of Generosity in each Region",color='r',fontsize=20)
plt.xlabel("Regions")
plt.ylabel("Generosity")
plt.xticks(rotation=90)
plt.show()

max_a = generosity[0]
min_a = generosity[0]

max_b = 0
min_b = 0

for i,g in enumerate(generosity):
    if min_a > generosity[i]:
        min_a = generosity[i]
        min_b = i
    elif max_a < generosity[i]:
        max_a = generosity[i]
        max_b = i
print("Least Generous Region:\n",region_names[min_b])
print("Most Generous Region:\n",region_names[max_b])

