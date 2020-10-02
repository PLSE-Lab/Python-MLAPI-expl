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
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


share_race_by_city = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv", encoding="windows-1252")
median_household_income_2015 = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv", encoding="windows-1252")
percent_over_25_completed_high_school = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv", encoding="windows-1252")
percentage_people_below_poverty_level = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv", encoding="windows-1252")
police_killings_us = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding="windows-1252")


# In[ ]:


# poverty rate
data = percentage_people_below_poverty_level
print(data.head(10))
print(data.describe())
print(data.info())


# In[ ]:


print(data.poverty_rate.value_counts())
data.poverty_rate.replace(["-"], 0.0, inplace=True)
data.poverty_rate = data.poverty_rate.astype("float")
print(data.poverty_rate.value_counts())
print(data.info())


# In[ ]:


area_list = list(data["Geographic Area"].unique())
print(area_list, len(area_list))

area_poverty_ratio = []
for i in area_list:
    x = data[data["Geographic Area"] == i]
    area_poverty_rate = sum(x.poverty_rate)/len(x.poverty_rate)
    area_poverty_ratio.append(area_poverty_rate)
    
data2 = pd.DataFrame({"area_list": area_list, "area_poverty_ratio": area_poverty_ratio})
print(data2)

new_index = (data2["area_poverty_ratio"].sort_values(ascending=False)).index.values
sorted_data = data2.reindex(new_index)
sorted_data


# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.barplot(x=sorted_data["area_list"], y=sorted_data["area_poverty_ratio"])
plt.xticks(rotation=90)
plt.xlabel("states")
plt.ylabel("poverty rate")
plt.title("poverty rate given states")
plt.show()


# In[ ]:


data = police_killings_us
data.head()
data.info()
data.name.value_counts(dropna=False)

_filter = data.name != "TK TK"
separate = data.name[_filter].str.split()
a,b = zip(*separate)
name_list = a+b
name_count = Counter(name_list)
most_common = name_count.most_common(15)
x,y = zip(*most_common)
x,y = list(x), list(y)

plt.figure(figsize=(15,10))
ax = sns.barplot(x=x, y=y, palette=sns.cubehelix_palette(len(x)))
plt.xlabel("name or surname of killed people")
plt.ylabel("Frequency")
plt.title("most common 15 name or surname of killed people")
plt.show()


# In[ ]:


data = percent_over_25_completed_high_school
data.info()
data["percent_completed_hs"].value_counts(dropna=False)
data["percent_completed_hs"].replace(["-"],0.0,inplace=True)
data["percent_completed_hs"] = data["percent_completed_hs"].astype(float)
data.info()
area_list = list(data["Geographic Area"].unique())
area_highschool = []
for i in area_list:
    _filter = data["Geographic Area"] == i
    x = data[_filter]
    area_highschool_rate = sum(x.percent_completed_hs)/len(x)
    area_highschool.append(area_highschool_rate)

data2 = pd.DataFrame({"area_list": area_list, "area_highschool": area_highschool})
new_index = data2.area_highschool.sort_values(ascending=True).index.values
sorted_data2 = data2.reindex(new_index)

plt.figure(figsize=(15,10))
ax=sns.barplot(x=sorted_data2["area_list"], y=sorted_data2["area_highschool"])
plt.xticks(rotation=90)
plt.xlabel("states")
plt.ylabel("high school graduation rate")
plt.title("bla bla bla")
plt.show()


# In[ ]:


data2.area_highschool[data2.area_list == "MA"]


# In[ ]:


data = share_race_by_city
data.head()
data.info()
data["City"].value_counts()

invalid_syntaxt_list = ["-", "(X)"]

for k in invalid_syntaxt_list:
    print("searching results for -> '{}'".format(k))
    for j in data.columns:
        counter = 0
        for i in data[j]:
            if i == k:
                counter += 1
        print(j,counter)
        
data.replace(["(X)"], 0.0, inplace=True)

# bunun yerine,
#data.share_white = data.share_white.astype(float)
#data.share_black = data.share_black.astype(float)
#data.share_native_american = data.share_native_american.astype(float)
#data.share_asian = data.share_asian.astype(float)
#data.share_hispanic = data.share_hispanic.astype(float)

# boyle yapabilirsin
data.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]] = data.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]].astype(float)

data.info()


# In[ ]:


area_list = list(data["Geographic area"].unique())

share_white = []
share_black = []
share_native_american = []
share_asian = []
share_hispanic = []

for i in area_list:
    _filter = data["Geographic area"]==i
    x = data[_filter]  
    _length = len(x)
    share_white.append(sum(x.share_white)/_length)
    share_black.append(sum(x.share_black)/_length)
    share_native_american.append(sum(x.share_native_american)/_length)
    share_asian.append(sum(x.share_asian)/_length)
    share_hispanic.append(sum(x.share_hispanic)/_length)


# In[ ]:


f, ax = plt.subplots(figsize=(9,15))

sns.barplot(x=share_white,y=area_list,color="green",alpha=0.5,label="white")
sns.barplot(x=share_black,y=area_list,color="blue",alpha=0.5,label="black")
sns.barplot(x=share_native_american,y=area_list,color="yellow",alpha=0.5,label="native")
sns.barplot(x=share_asian,y=area_list,color="red",alpha=0.5,label="asian")
sns.barplot(x=share_hispanic,y=area_list,color="cyan",alpha=0.5,label="hispanic")

ax.legend(loc="lowerright", frameon=True)
ax.set(xlabel="percentage of races",ylabel="states",title="bla bla")
plt.show()


# In[ ]:


sorted_data["area_poverty_ratio"] = sorted_data["area_poverty_ratio"]/max(sorted_data["area_poverty_ratio"])
sorted_data2["area_highschool"] = sorted_data2["area_highschool"]/max(sorted_data2["area_highschool"])
data = pd.concat([sorted_data, sorted_data2["area_highschool"]],axis=1)
data.sort_values("area_poverty_ratio",inplace=True)

f, ax1 = plt.subplots(figsize=(20,10))
sns.pointplot(x="area_list",y="area_poverty_ratio",data=data,color="lime",alpha=0.8)
sns.pointplot(x="area_list",y="area_highschool",data=data,color="red",alpha=0.8)
plt.text(40,0.6,"asdasd",color="red",fontsize=17,style="italic")
plt.text(40,0.55,"qweqwe",color="lime",fontsize=17,style="italic")
plt.xlabel("states",fontsize=20,color="blue")
plt.ylabel("values",fontsize=20,color="blue")
plt.grid()
plt.show()


# In[ ]:


data.head()
g=sns.jointplot(data.area_poverty_ratio, data.area_highschool, kind="kde", height=7)
plt.show()


# In[ ]:


g=sns.jointplot("area_poverty_ratio", "area_highschool", data=data, color="red", ratio=3, height=10)
plt.show()


# In[ ]:


data3=police_killings_us
data3.head()
data3.race.value_counts(dropna=False)
data3.race.dropna(inplace=True)
labels = data3.race.value_counts().index
sizes = data3.race.value_counts().values
colors = ["green","yellow","red","blue","cyan","magenta"]
explode = [0.1,0.1,0.1,0.1,0.1,0.1]

plt.figure(figsize=(8,8))
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct="%1.1f%%")
plt.title("title",color="blue",fontsize=15)
plt.show()


# In[ ]:


data.head()
plt.figure(figsize=(12,12))
sns.lmplot(x="area_poverty_ratio",y="area_highschool",data=data)
plt.show()


# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(15,6))
sns.kdeplot(data.area_poverty_ratio,data.area_highschool,shade=True,cut=5,ax=axes[0])
sns.kdeplot(data.area_poverty_ratio,data.area_highschool,shade=False,cut=5,ax=axes[1])

plt.show()


# In[ ]:


pal=sns.cubehelix_palette(2,rot=-0.5,dark=0.3)
sns.violinplot(data=data,palette=pal,inner="points")
plt.show()


# In[ ]:


f,ax=plt.subplots(figsize=(6,6))
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt=".1f",ax=ax)

plt.show()


# In[ ]:


f,ax=plt.subplots(figsize=(6,6))
sns.boxplot(x="manner_of_death",y="age",hue="gender",data=data3,palette="PRGn")
plt.show()


# In[ ]:


f,ax=plt.subplots(figsize=(6,6))
sns.swarmplot(x="gender",y="age",hue="manner_of_death",data=data3,)
plt.show()


# In[ ]:


sns.pairplot(data)
plt.show()


# In[ ]:


sns.countplot(data3.race)
plt.show()


# In[ ]:


armed = data3.armed.value_counts()
sns.barplot(x=armed[:7].index,y=armed[:7].values)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


above25 = ["above25" if age>=25 else "below25" for age in data3.age]
df = pd.DataFrame({"age": above25})
sns.countplot(x=df.age)
plt.show()


# In[ ]:


state = data3.state.value_counts()
sns.barplot(x=state[:15].index, y=state[:15].values)
plt.show()

