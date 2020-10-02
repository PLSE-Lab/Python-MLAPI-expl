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
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


MedianHouseholdIncome=pd.read_csv('../input/MedianHouseholdIncome2015.csv',encoding='windows-1252')
PercentOver25CompletedHighSchool=pd.read_csv('../input/PercentOver25CompletedHighSchool.csv',encoding='windows-1252')
ShareRaceByCity=pd.read_csv('../input/ShareRaceByCity.csv',encoding='windows-1252')
PoliceKillingsUS=pd.read_csv('../input/PoliceKillingsUS.csv',encoding='windows-1252')
PercentagePeopleBelowPovertyLevel=pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv',encoding='windows-1252')


# In[ ]:


PercentagePeopleBelowPovertyLevel.head()


# In[ ]:


PercentagePeopleBelowPovertyLevel.poverty_rate.value_counts()


# In[ ]:


PercentagePeopleBelowPovertyLevel.poverty_rate.replace(['-'],0.0,inplace=True)


# In[ ]:


PercentagePeopleBelowPovertyLevel.poverty_rate=PercentagePeopleBelowPovertyLevel.poverty_rate.astype(float)


# In[ ]:


type(PercentagePeopleBelowPovertyLevel.poverty_rate)


# In[ ]:


area_list=list(PercentagePeopleBelowPovertyLevel['Geographic Area'].unique())


# In[ ]:


area_poverty_ratio=[]
for i in area_list:
    x=PercentagePeopleBelowPovertyLevel[PercentagePeopleBelowPovertyLevel['Geographic Area']==i]
    area_poverty_rate=sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)


data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})
new_index=(data['area_poverty_ratio'].sort_values(ascending=False)).index.values
sorted_data=data.reindex(new_index)

plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])
plt.xticks(rotation= 45)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given States')


# In[ ]:


PoliceKillingsUS.head()


# In[ ]:


separate = PoliceKillingsUS.name[PoliceKillingsUS.name != 'TK TK'].str.split() 
a,b = zip(*separate)
name_list = a+b                         
name_count = Counter(name_list)         
most_common_names = name_count.most_common(15)  
x,y = zip(*most_common_names)
x,y = list(x),list(y)
# 
plt.figure(figsize=(15,10))
ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Name or Surname of killed people')
plt.ylabel('Frequency')
plt.title('Most common 15 Name or Surname of killed people')


# In[ ]:


PercentOver25CompletedHighSchool.head()


# In[ ]:


PercentOver25CompletedHighSchool.percent_completed_hs.value_counts()


# In[ ]:


PercentOver25CompletedHighSchool.info()


# In[ ]:


PercentOver25CompletedHighSchool.percent_completed_hs.replace(['-',0.0],inplace=True)


# In[ ]:


PercentOver25CompletedHighSchool.percent_completed_hs=PercentOver25CompletedHighSchool.percent_completed_hs.astype(float)


# In[ ]:


area_list=list(PercentOver25CompletedHighSchool['Geographic Area'].unique())


# In[ ]:


area_highschool=[]
for i in area_list:
    x = PercentOver25CompletedHighSchool[PercentOver25CompletedHighSchool['Geographic Area']==i]
    area_highschool_rate = sum(x.percent_completed_hs)/len(x)
    area_highschool.append(area_highschool_rate)
data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool})
new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values
sorted_data2 = data.reindex(new_index)


# In[ ]:


# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])
plt.xticks(rotation= 90)
plt.xlabel('States')
plt.ylabel('High School Graduate Rate')
plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")


# In[ ]:


ShareRaceByCity.head()


# In[ ]:


ShareRaceByCity.replace(['(X)'],0.0,inplace=True)


# In[ ]:


ShareRaceByCity.share_white=ShareRaceByCity.share_white.astype(float)
ShareRaceByCity.share_black=ShareRaceByCity.share_black.astype(float)
ShareRaceByCity.share_native_american=ShareRaceByCity.share_native_american.astype(float)
ShareRaceByCity.share_asian=ShareRaceByCity.share_asian.astype(float)
ShareRaceByCity.share_hispanic=ShareRaceByCity.share_hispanic.astype(float)

ShareRaceByCity.info(0)


# In[ ]:


area_list=list(ShareRaceByCity['Geographic area'].unique())


# In[ ]:


white_p=[]
black_p=[]
american_p=[]
asian_p=[]
hispanic_p=[]
for i in area_list:
    x=ShareRaceByCity[ShareRaceByCity['Geographic area']==i]
    white_p.append(sum(x.share_white)/len(x))
    black_p.append(sum(x.share_black)/len(x))
    american_p.append(sum(x.share_native_american)/len(x))
    asian_p.append(sum(x.share_asian)/len(x))
    hispanic_p.append(sum(x.share_hispanic)/len(x))

f,ax = plt.subplots(figsize = (9,15))
sns.barplot(x=white_p,y=area_list,color='green',alpha = 0.5,label='White' )
sns.barplot(x=black_p,y=area_list,color='b',alpha = 0.5,label='Black' )
sns.barplot(x=american_p,y=area_list,color='red',alpha = 0.5,label='Native American' )
sns.barplot(x=asian_p,y=area_list,color='pink',alpha = 0.5,label='Asian' )
sns.barplot(x=american_p,y=area_list,color='orange',alpha = 0.5,label='Hispanic' )
ax.legend(loc='upper right',frameon = True) 
ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")


# In[ ]:


sorted_data.info()


# In[ ]:


# high school graduation rate vs Poverty rate of each state
sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])
sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])


# In[ ]:


data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)


# In[ ]:


data.head()


# In[ ]:


data.sort_values('area_poverty_ratio',ascending=False,inplace=True)


# In[ ]:


f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8)
sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='red',alpha=0.8)
plt.grid()


# In[ ]:


g=sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio,kind='kde',size=7)
plt.savefig('graph.png')
plt.show()


# In[ ]:


g=sns.jointplot('area_poverty_ratio','area_highschool_ratio',data=data,size=5,ratio=3,color='r')


# In[ ]:


PoliceKillingsUS.head()


# In[ ]:


sizes=PoliceKillingsUS.race.value_counts().values
labels=PoliceKillingsUS.race.value_counts().index
colors=['red','blue','orange','green','yellow','pink']
explode=[0,0,0,0,0,0]

plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode,labels=labels,colors=colors, autopct='%1.1f%%')


# In[ ]:


sns.lmplot(x='area_poverty_ratio', y='area_highschool_ratio', data=data)


# In[ ]:


sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade=True, cut=3)
plt.show()


# In[ ]:


pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=data, palette=pal, inner="points")
plt.show()


# In[ ]:


data.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax)
plt.show()


# In[ ]:


data.head()


# In[ ]:


PoliceKillingsUS.head()


# In[ ]:


sns.boxplot(x='gender',y='age',hue='manner_of_death',data=PoliceKillingsUS,palette='PRGn')


# In[ ]:


# swarm plot
# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla
# gender cinsiyet
# age: yas
sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=PoliceKillingsUS)
plt.show()


# In[ ]:


sns.pairplot(data)


# In[ ]:


PoliceKillingsUS.head()


# In[ ]:


sns.countplot(x='gender',data=PoliceKillingsUS)
sns.countplot(x='manner_of_death',data=PoliceKillingsUS)


# In[ ]:


armed=PoliceKillingsUS.armed.value_counts()


# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(x=armed[0:7].index,y=armed[0:7].values)


# In[ ]:


above25 =['above25' if i >= 25 else 'below25' for i in PoliceKillingsUS.age]
df=pd.DataFrame({'age':above25})
sns.countplot(x=df.age)


# In[ ]:


city=PoliceKillingsUS.city.value_counts()
plt.figure(figsize=(17,8))
sns.barplot(x=city[:12].index,y=city[:12].values)


# In[ ]:




