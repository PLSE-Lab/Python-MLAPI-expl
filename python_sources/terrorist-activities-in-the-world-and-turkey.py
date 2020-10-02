#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Hello world. I'm just beginning with data science. I chose the data review this reason i want to explain the situation in Turkey this activity I have been living for years.There are many terrorist activities in my country.I hope you guys can't experience this. Started.

# In[ ]:


data_completed = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')


# In[ ]:


#The names of the columns in the data are somewhat irregular.


data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)

data=data[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]


data['casualities']=data['Killed']+data['Wounded']


# In[ ]:


#Now we can check the columns.
data.columns


# In[ ]:


#Let's try to get an idea about the first five columns
data.head()


# In[ ]:


#Let's see the columns.
data.info()


# In[ ]:


#Examine the graph that gives the ratio between the columns.

data.corr()


# In[ ]:


#Let's look at an asdas map.
#corelation map

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In this way we can see the proportion between the pillars more easily.For example with the year of latitude as the rate...
#     Although there are rates that we cannot understand, they can give us a general idea.
#     
#     Now some check in the matplot..

# In[ ]:


#MATPLOTLIB
#Although the number of wounded curve is large, the total number of people who died in total is very high.
# Line Plot
#Wounded And Killed Graphics
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Wounded.plot(kind = 'line', color = 'g',label = 'Wounded',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Killed.plot(color = 'r',label = 'Killed',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
#Wounded And Killed
# x = Wounded, y = Killed
data.plot(kind='scatter', x='Wounded', y='Killed',alpha = 0.5,color = 'red')
plt.xlabel('Wounded')              # label = name of label
plt.ylabel('Killed')
plt.title('Killed Wounded Scatter Plot')            # title = title of plot
plt.show()


# According to year increase of terrorist activities all over the world and turkey she was deeply touched.
# Check for Graphics...
# 

# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot('Year',data=data,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# In[ ]:


#Different graphics
#%%


# Scatter Plot 
#Year And Killed
# x = yeaer, y = killed
data.plot(kind='scatter', x='Year', y='Killed',alpha = 0.5,color = 'red')
plt.xlabel('Year')              # label = name of label
plt.ylabel('Killed')
plt.title('Killed Wounded Scatter Plot')            # title = title of plot


# In[ ]:


# Histogram
#This is year histogram graphics,this graph will give us the frequency of terrorist activities over the years
# bins = number of bar in figure
data.Year.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:


# Only last year is the terror activities
# 1 - Filtering Pandas data frame
year = data['Year']>2016     
data[year]


# In[ ]:


#Only terrorist activities in Turkey
#Let's look at the first 20
data_tr = data_completed[data_completed.country_txt == 'Turkey']
data_tr.head(20)


# In[ ]:


#Only the number of terrorist attacks in Turkey
data_tr.iyear.value_counts(dropna = False).sort_index()


# Compared to Earth, we compare the number of attacks every year.The number of attacks compared to the real world, as seen in turkey You seem to be too much.I hope you are not so much in your country.

# In[ ]:



data_tr.iyear.plot(kind = 'hist', color = 'r', bins=range(1970, 2018), figsize = (16,7), alpha=0.5, grid=True)
plt.xticks(range(1970, 2018), rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Years", fontsize=15)
plt.ylabel("Number of Attacks", fontsize=15)
plt.title("Only Turkey Number of Attacks By Year", fontsize=16)
plt.show()


data.Year.plot(kind = 'hist', color = 'b', bins=range(1970, 2018), figsize = (16,7), alpha=0.5, grid=True)
plt.xticks(range(1970, 2018), rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Years", fontsize=15)
plt.ylabel("Number of Attacks", fontsize=15)
plt.title("Number of Attacks By Year", fontsize=16)
plt.show()


# **Now let us examine the total number of terrorist acts and dead numbers in the world.**
# 
# **Compare the turkey with them**

# In[ ]:


data_completed = data_completed[np.isfinite(data_completed.latitude)]

count_year = data_completed.groupby(['iyear']).count()
mean_year = data_completed.groupby(['iyear']).mean()

fig = plt.figure(figsize = (10,8))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.set(title = 'Total acts of terrorism', ylabel = 'Act Count', xlabel = 'Year')
ax1.plot(count_year.index, count_year.eventid)
ax2.set(title = 'Average Number of Deaths per Act', ylabel = 'Death Count', xlabel = 'Year')
ax2.plot(mean_year.index, mean_year.nkill)
fig.autofmt_xdate()


count_year = data_tr.groupby(['iyear']).count()
mean_year = data_tr.groupby(['iyear']).mean()

fig = plt.figure(figsize = (10,8))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.set(title = 'Total acts of terrorism in the Turkey', ylabel = 'Act Count', xlabel = 'Year')
ax1.plot(count_year.index, count_year.eventid)
ax2.set(title = 'Average Number of Deaths per Act in the Turkey', ylabel = 'Death Count', xlabel = 'Year')
ax2.plot(mean_year.index, mean_year.nkill)
fig.autofmt_xdate()


# Below are terrorist groups and actors.The result is more in the groups where the population is higher.As the number of people increases, terror activities are increasing at the same rate.

# In[ ]:


#Terorism actors and groups in the global
terorism_actors = data_completed.gname.unique()
print(terorism_actors)

attacks_of_groups = []
for name in terorism_actors:
    temp = data_completed.gname[data_completed.gname == name].count()
    attacks_of_groups.append(temp)
    
dataframe_temp = pd.DataFrame({'actor':terorism_actors, 'attack_num':attacks_of_groups})
#data is very long for attacks in mt cut 
#Attacks greater than 6
dataframe_temp = dataframe_temp[dataframe_temp.attack_num >= 6]
dataframe_temp


# In[ ]:


#Terorism actors and groups in the turkey
terorism_actors_tr = data_tr.gname.unique()
print(terorism_actors_tr)

attacks_of_groups_tr = []
for name in terorism_actors_tr:
    temp = data_tr.gname[data_tr.gname == name].count()
    attacks_of_groups_tr.append(temp)
    
dataframe_temp_tr = pd.DataFrame({'actor':terorism_actors_tr, 'attack_num':attacks_of_groups_tr})
#Attacks greater than 6
dataframe_temp_tr = dataframe_temp_tr[dataframe_temp_tr.attack_num >= 6]
dataframe_temp_tr


# **The above graphs and numbers show how much terror has affected the world.I compared it to my home country as much as I could and made short comments on the results.I leave the original comment to your esteemed members.**
# 
# ***Hope it has been a useful review***
# 
# 
# **Respects...**

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




