#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# * **In this kernel, we will learn how to use seaborn library.**

# **Content**
# 
# * [Loading the Data Set](#1)
# * [Bar Plot](#2)
# * [Point Plot](#3)
# * [Joint Plot](#4)
# * [Pie Chart](#5)
# * [Lm Plot](#6)
# * [Kde Plot](#7)
# * [Box Plot](#8)
# * [Swarm Plot](#9)
# * [Pair Plot](#10)
# * [Count Plot](#11)

# <a id="1"></a> <br>
# **Loading the Data Set**
# * *Reading datas*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Read datas
MedianHouseholdIncome2015 = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")
PercentagePeopleBelowPovertyLevel = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")
PercentOver25CompletedHighSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")
ShareRaceByCity = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")
PoliceKillingsUS = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")


# In[ ]:


PercentagePeopleBelowPovertyLevel.info()


# In[ ]:


PercentagePeopleBelowPovertyLevel.poverty_rate.value_counts()


# <a id="2"></a> <br>
# **Bar Plot**

# In[ ]:


#poverty rate of each state
PercentagePeopleBelowPovertyLevel.poverty_rate.replace(['-'],0.0,inplace=True)
PercentagePeopleBelowPovertyLevel.poverty_rate = PercentagePeopleBelowPovertyLevel.poverty_rate .astype("float") 
area_list = list(PercentagePeopleBelowPovertyLevel["Geographic Area"].unique())
#print(len(area_list))
area_poverty_ratio= []

for i in area_list:
    x = PercentagePeopleBelowPovertyLevel[PercentagePeopleBelowPovertyLevel["Geographic Area"]==i]
    area_poverty_rate = sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)
    
data = pd.DataFrame({'area_list':area_list,'area_poverty_ratio':area_poverty_ratio})
new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)


#visualizaiton

plt.figure(figsize=(15,10)) 
sns.barplot(x=sorted_data["area_list"], y = sorted_data["area_poverty_ratio"])
plt.xticks(rotation = 90)
plt.xlabel("States")
plt.ylabel("Poverty Rate")
plt.title("Poverty Rate Given States")


# In[ ]:


PoliceKillingsUS.head()


# In[ ]:


# Most common 15 Name or Surname of killed people
seperate = PoliceKillingsUS.name[PoliceKillingsUS.name != 'TK TK'].str.split()
name1,name2 = zip(*seperate) 
name_list = name1+name2
name_count  = Counter(name_list)
mostCommonName = name_count.most_common(15)
x,y = zip(*mostCommonName)
x,y = list(x),list(y)

plt.figure(figsize=(15,10))
ax = sns.barplot(x=x,y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel("Name or surname")
plt.ylabel("Fruquencey")
plt.title("Most Common 15 Name or Surname")


# In[ ]:


PercentOver25CompletedHighSchool.head()


# In[ ]:


PercentOver25CompletedHighSchool.percent_completed_hs.value_counts()


# **High school graduation rate vs Poverty rate of each state**

# In[ ]:


#Graduate rate of each state
PercentOver25CompletedHighSchool.percent_completed_hs.replace(['-'],0.0,inplace=True)
PercentOver25CompletedHighSchool.percent_completed_hs = PercentOver25CompletedHighSchool.percent_completed_hs .astype("float") 
area_list = list(PercentOver25CompletedHighSchool["Geographic Area"].unique())
#print(len(area_list))
area_highschool_ratio = []

for i in area_list:
    x = PercentOver25CompletedHighSchool[PercentOver25CompletedHighSchool["Geographic Area"]==i]
    area_highschool_rate = sum(x.percent_completed_hs)/len(x)
    area_highschool_ratio.append(area_highschool_rate) 
    
data = pd.DataFrame({'area_list':area_list,'area_highschool_ratio':area_highschool_ratio})
new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values
sorted_data2 = data.reindex(new_index)


#visualizaiton

plt.figure(figsize=(15,10)) 
sns.barplot(x=sorted_data2["area_list"], y = sorted_data2["area_highschool_ratio"])
plt.xticks(rotation = 90)
plt.xlabel("States")
plt.ylabel("High School Graduate Rate")
plt.title("Graduate Rate Given States")


# In[ ]:


ShareRaceByCity.info()


# In[ ]:


ShareRaceByCity.replace(["-"],0.0,inplace=True)
ShareRaceByCity.replace(["(X)"],0.0,inplace=True)
ShareRaceByCity.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = ShareRaceByCity.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)  
area_list = list(ShareRaceByCity["Geographic area"].unique())
share_white = []
share_black = []
share_native_american = []
share_asian = []
share_hispanic = []

cont = 0;

for i in area_list:
    cont = cont+1
    x = ShareRaceByCity[ShareRaceByCity['Geographic area'] == i]
    share_white.append(sum(x.share_white) / len(x))
    share_black.append(sum(x.share_black) / len(x))
    share_native_american.append(sum(share_native_american) / len(x))
    share_asian.append(sum(x.share_asian) / len(x))
    share_hispanic.append(sum(x.share_hispanic) / len(x))
    
f,ax = plt.subplots(figsize=(9,15))
sns.barplot(x=share_white, y = area_list, color="green",alpha = 0.5,label = "White")
sns.barplot(x=share_black, y = area_list, color="blue",alpha = 0.7,label = "African American")
sns.barplot(x=share_native_american, y = area_list, color="cyan",alpha = 0.6,label = "Native American")
sns.barplot(x=share_asian, y = area_list, color="red",alpha = 0.6,label = "Asian")
sns.barplot(x=share_hispanic, y = area_list, color="yellow",alpha = 0.6,label = "Hispanic")

ax.legend(loc="lower right",frameon=True)
ax.set(xlabel="Percentage of Races",ylabel="States",title="Percentage of State's Population According To Races")


# <a id="3"></a> <br>
# **Point Plot**

# In[ ]:


# high school graduation ratio VS poverty rate of each state
sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio'] / max(sorted_data['area_poverty_ratio'])
sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio'] / max(sorted_data2['area_highschool_ratio'])
data_Concate = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1) 
data_Concate.sort_values('area_poverty_ratio',inplace=True)

f,ax = plt.subplots(figsize=(20,10))
sns.pointplot(x='area_list',y='area_poverty_ratio',data=data_Concate,color='lime',alpha=0.8)
sns.pointplot(x='area_list',y='area_highschool_ratio',data=data_Concate,color='red',alpha=0.8)
plt.text(40,0.6,'high school graduate ratio',color='red',fontsize=17,style='italic')
plt.text(40,0.55,'poverty ratio',color='lime',fontsize=18,style='italic')
plt.xlabel ('States',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('High School Graduate VS Poverty Rate',fontsize = 20, color='blue')
plt.grid()


# In[ ]:


data_Concate.head()


# <a id="4"></a> <br>
# **Joint Plot**

# In[ ]:


#pearsonr = correlation between variables
#if it is 1 , there is positive correlation , if it is -1 , there is negative correlation
#if it is 0 , there is no correlation
#pdf probability density function
joint = sns.jointplot(data_Concate.area_poverty_ratio , data_Concate.area_highschool_ratio, kind = "kde",size=7)
plt.savefig("graph.png")
plt.show()


# In[ ]:


joint = sns.jointplot("area_poverty_ratio","area_highschool_ratio",data = data_Concate ,size = 5 ,ratio = 3, color="r" )


# <a id="5"></a> <br>
# **Pie Chart**

# In[ ]:


PoliceKillingsUS.race.head(15)


# In[ ]:


PoliceKillingsUS.race.dropna(inplace=True)
labels = PoliceKillingsUS.race.value_counts().index
colors = ["grey","blue","red","yellow","green","brown"]
explode = [0,0,0,0,0,0]
sizes = PoliceKillingsUS.race.value_counts().values

plt.figure(figsize=(7,7))
plt.pie(sizes,explode = explode, labels=labels,colors=colors,autopct='%1.1f%%')
plt.title("Killed people According to Races",color="blue",fontsize=15)


# <a id="6"></a> <br>
# **Lm Plot**

# In[ ]:


sns.lmplot("area_poverty_ratio","area_highschool_ratio",data = data_Concate)
plt.show()


# In[ ]:


sns.kdeplot(data_Concate.area_poverty_ratio,data_Concate.area_highschool_ratio, shade=True,cut=5)
plt.show()


# In[ ]:


data_Concate.corr()


# <a id="7"></a> <br>
# **Kde Plot**

# In[ ]:


#correlation map - heat map
f,ax = plt.subplots(figsize=(5,5))
sns.heatmap(data_Concate.corr(),annot = True,linewidths = .5,linecolor="red",fmt = '.1f',ax=ax)
plt.show()


# <a id="8"></a> <br>
# **Box Plot**

# In[ ]:


PoliceKillingsUS.head()


# In[ ]:


#Manner of Deadth 
plt.figure(figsize=(10,7))
sns.boxplot(x="gender",y="age",hue="manner_of_death",data=PoliceKillingsUS,palette="PRGn")


# <a id="9"></a> <br>
# **Swarm Plot**

# In[ ]:


plt.figure(figsize=(8,5))
sns.swarmplot(x="gender",y="age",hue="manner_of_death",data=PoliceKillingsUS)
plt.show()


# <a id="10"></a> <br>
# **Pair Plot**

# In[ ]:


#pair plot
sns.pairplot(data_Concate)
plt.show()


# <a id="11"></a> <br>
# **Count Plot**

# In[ ]:


PoliceKillingsUS.manner_of_death.value_counts()


# In[ ]:


sns.countplot(PoliceKillingsUS.gender)
#sns.countplot(PoliceKillingsUS.manner_of_death)
plt.title("gender",color="blue",fontsize=15)
plt.show()


# In[ ]:


armed = PoliceKillingsUS.armed.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=armed[:7].index,y = armed[:7].values)
plt.ylabel("Number of Weapon")
plt.xlabel("Weapon Types")
plt.title("Kill Weapon",color="red",fontsize=20)


# In[ ]:


above25 = ["above25" if i>25 else "belov25" for i in PoliceKillingsUS.age]
datafrm = pd.DataFrame({'age':above25})
sns.countplot(x=datafrm.age)
plt.ylabel("Number Of Killed People")
plt.title("Age Of Killed people",color="blue",fontsize=15)


# In[ ]:


sns.countplot(data=PoliceKillingsUS,x="race")
plt.title("Race of killed people",color="blue",fontsize=15)


# In[ ]:


city = PoliceKillingsUS.city.value_counts()
plt.figure(figsize=(12,8))
sns.barplot(x=city[:12].index,y=city[:12].values)
plt.xticks(rotation=45)
plt.title("Most Dangerous cities",color="green",fontsize=20)

