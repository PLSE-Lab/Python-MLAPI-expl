#!/usr/bin/env python
# coding: utf-8

# **Before start up I would like to thank you [DATAI Team](http://https://www.kaggle.com/kanncaa1) for their help.**

# This Tutorial is my second tutorial for Visualization of [Seaborn Galery](http://https://seaborn.pydata.org/examples/index.html). Iwould like to show you what I learn and I want to practicer.
# If you want to see my first one. It is [HERE](http://https://www.kaggle.com/aliylmaz0907/suicide-statistics-with-seaborn-library-1)
# 
# 
# # INTRODUCTION
# 1. Read data
# 1. Reaname the columns 
# 1. For each state Victim numbers
# 1. For each state Perpetrator  numbers
# 1. Victim's race
# 1. Weapons types
# 1. Each state most commun race
# <br>
# <br>
# Plot Contents:
# * [Bar Plot](#1)
# * [[Pie Chart](#2)
# * [Joint Plot ](#3)
# * [Joint Plot HEX](#4)
# * [Joint Plot SCATTER](#5)
# * [Kde Plot](#6)
# * [Count Plot](#7)
# * [Heatmap](#8)
# * [Implot](#9)
# * [Box Plot](#10)
# * [RelPlot](#11)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/database.csv')


# In[ ]:


data.tail()


# In[ ]:


data.rename(columns={'Crime Type':'Crime_Type','Victim Sex':'V_Sex','Victim Age':'V_Age','Victim Race':'V_Race','Perpetrator Race':'P_race','Perpetrator Age':'P_Age','Perpetrator Sex':'P_Sex','Victim Count':'V_Count','Perpetrator Count':'P_Count'},inplace=True)


# In[ ]:


data.head()


# <a id="1"></a> 
# ## Bar Plot

# In[ ]:


state_list=list(data['State'].unique())
victim=[]
perpetrator=[] 


for i in state_list:
    x=data[data.State==i]
    victim.append(sum(x.V_Count))
    perpetrator.append(sum(x.P_Count))
    
datanew=pd.DataFrame({'state_list':state_list,'victim_nu':victim,'perpetrator':perpetrator})
new_index = (datanew['victim_nu'].sort_values(ascending=False)).index.values
sorted_data = datanew.reindex(new_index)

# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['state_list'], y=sorted_data['victim_nu'])
plt.xticks(rotation= 90)
plt.xlabel('States')
plt.ylabel('Victim Numbers')
plt.title('Victim Numbers Given States')


# In[ ]:


datanew.info()


# <a id="2"></a> 
# ## Pie Chart

# In[ ]:


data.V_Race.dropna(inplace = True)
labels = data.V_Race.value_counts().index
colors = ['grey','blue','red','yellow','green','brown']
explode = [0,0,0,0,0,0]
sizes = data.V_Race.value_counts().values


plt.figure(figsize = (7,7))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Victims According to Races',color = 'blue',fontsize = 15)


# In[ ]:


data.head()


# In[ ]:


data.Weapon.dropna(inplace = True)
labels = data.Weapon.value_counts().index
colors = ['grey','blue','red','yellow','green','brown']
explode = [0,0,0,0,0,0]
sizes = data.Weapon.value_counts().values


plt.figure(figsize = (7,7))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Weapons types',color = 'blue',fontsize = 15)


# In[ ]:


data.head()


# In[ ]:


state_list=list(data['State'].unique())
v_race_commun=[]
p_race_commun=[]
for i in state_list:
    x=data[data['State']==i]
    r_count = Counter(x.V_Race)
    most_com = r_count.most_common(1)
    v_race_commun.append(most_com)
    p_count = Counter(x.P_race)
    cost_com = p_count.most_common(1)
    p_race_commun.append(cost_com)
    
datanew['most_vic_race']=v_race_commun
datanew['most_per_race']=p_race_commun


# In[ ]:


datanew.head(25)


# <a id="3"></a> 
# ## Joint Plot

# In[ ]:


g = sns.jointplot(datanew.victim_nu, datanew.perpetrator, kind="kde", size=10)
plt.savefig('graph.png')
plt.show()


# In[ ]:


datanew.head()


# <a id="4"></a> 
# ## Joint Plot HEX

# In[ ]:


sns.jointplot("victim_nu", "perpetrator",data=datanew, kind="hex", color="#4CB391")


# <a id="5"></a> 
# ## Joint Plot SCATTER

# In[ ]:


g = sns.jointplot("victim_nu", "perpetrator", data=datanew,size=5, ratio=3, color="r")


# <a id="6"></a> 
# ## Kde Plot

# In[ ]:


sns.kdeplot(datanew.victim_nu, datanew.perpetrator, shade=True, cut=3)
plt.show()


# In[ ]:


data.info()


# <a id="7"></a> 
# ## Count Plot

# In[ ]:


sns.set_style('whitegrid')

sns.countplot(x='V_Sex', data= data)


# In[ ]:


sns.set_style('whitegrid')

sns.countplot(x='P_Sex', data= data)


# In[ ]:


data.head()


# In[ ]:


sns.countplot(x='Crime_Type', hue='P_Sex', data= data,palette='RdBu_r')


# <a id="8"></a> 
# ## Heat Map

# In[ ]:


f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


data.P_Age.value_counts()


# In[ ]:


data["P_Age"] = pd.to_numeric(data["P_Age"], errors='coerce')


# In[ ]:


data.info()


# In[ ]:


datanew.head()


# <a id="9"></a> 
# ## ImPlot

# In[ ]:


sns.lmplot(x="victim_nu", y="perpetrator", data=datanew)
plt.show()


# <a id="10"></a> 
# ## Box Plot

# In[ ]:


sns.boxplot(x="V_Sex", y="V_Age", hue="Crime_Type", data=data, palette="PRGn")
plt.show()


# In[ ]:


data.head()


# <a id="11"></a> 
# ## RelPlot

# In[ ]:


g=sns.relplot(x="State", y="Year", hue="V_Sex", size="V_Age",
            sizes=(0, 100), alpha=.5, palette="muted",
            height=6, data=data)
g.set_xticklabels(rotation=90)


# If you have a problem please tell me :)
