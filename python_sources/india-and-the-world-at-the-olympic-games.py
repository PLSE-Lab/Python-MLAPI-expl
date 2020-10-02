#!/usr/bin/env python
# coding: utf-8

# This is my first project on Kaggle.It took me some time to figure out how to use kaggle.Thinks like importing data and submitting a kernel.I feel happy that I could finally publish this kernel.This is still a work in process.I like history so doing a project on History on OLympic games was a learning process.I tried to get information on Indias performance at the Olympic games.If you like this please do vote for me by clicking at the vote option at the top.

# In[ ]:


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


# # Video on the history of Olympic games

# In[ ]:


from IPython.display import YouTubeVideo

YouTubeVideo('IUccWy-WzuA', width=800, height=450)


# # Five rings in the Olympic symbol represents five continents  

# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
img=np.array(Image.open('../input/olympic-symbol/OLympic_1.png'))
fig=plt.figure(figsize=(10,10))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
data=pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
data.head()


# # Finding out the years in which Games were held

# In[ ]:


print('Summer Olympics were held in the years:',np.array(sorted(data[data['Season']=='Summer']['Year'].unique())))


# In[ ]:


print('Winter Olympics were held in the years:',np.array(sorted(data[data['Season']=='Winter']['Year'].unique())))


# # Finding out the cities that have hosted Games 

# In[ ]:


Cities=data.groupby('City').apply(lambda x:x['Year'].unique()).to_frame().reset_index()
Cities.columns=['City','Years']
Cities['Occurence']=[len(c) for c in Cities['Years']]
Cities.sort_values('Occurence',ascending=False)


# # Getting the correlation between the Age,Weight and Height of Olympic athletes

# 

# In[ ]:


data[['Age','Height','Weight']].corr()


# # Finding out participation of Men and Women at the Olympic games

# In[ ]:


print('Total number of athletes to have taken part in Olympic games--->',len(data.ID.unique()))


# In[ ]:


print('Number of female participants at the Games--->',len(data[data.Sex=='F']))
print('Number of male participants at the Games--->',len(data[data.Sex=='M']))


# Some of the athletes participate in more than one game, this leads to duplication of male and female participation count.

# In[ ]:


import seaborn as sns
data.Sex.unique()
data.Sex.value_counts()
sns.countplot(data.Sex)
plt.title('Male Female participation in Olympic',size=15,color='green')
plt.show()


# # Participation of Men and Women in the Games over 120 years

# In[ ]:


part = data.groupby('Year')['Sex'].value_counts()
part.loc[:,'F'].plot(title='Men/Women participation in Olympic',figsize=(13,5)).set_ylabel("Athletes")
part.loc[:,'M'].plot()


# # Finding out the Youngest,Oldest and the Median age of athletes to have participated in the games
# 

# In[ ]:


print('The youngest athlete:',data.Age.min())
print('The average age of athletes:',data.Age.mean())
print('The oldest athlete:',data.Age.max())


# In[ ]:


y=np.array([data.Age.min(),data.Age.mean(),data.Age.max()])
x=['Youngest','Average','Oldest']
plt.bar(x,y)
plt.xlabel('Feature')
plt.ylabel('Age')
plt.show()


# # Plotting Height and Weight data of the athletes 

# In[ ]:


x=sns.distplot(data['Age'].dropna(),color='Red',kde=True)
x.set_title('Age Distribution of Athletes',fontsize=16,fontweight=200)


# In[ ]:


h=sns.distplot(data['Height'].dropna(),color='Green',kde=True)
h.set_title('Height Distribution of Athletes',fontsize=16,fontweight=200)


# In[ ]:


w=sns.distplot(data['Weight'].dropna(),color='Blue',kde=True)
w.set_title('Weight Distribution of Athletes',fontsize=16,fontweight=200)


# In[ ]:


f,ax=plt.subplots(figsize=(20,10))
sns.distplot(data['Age'].dropna(),color='Red',kde=True)
sns.distplot(data['Height'].dropna(),color='Green',kde=True)
sns.distplot(data['Weight'].dropna(),color='Blue',kde=True)


# # Finding out the five top nations in 120 years of Olympic games 

# In[ ]:


data.Medal.value_counts()


# Above numbers give data of total gold,silver and bronze medals won by atheletes at the OLympic Games

# In[ ]:


plt.subplot(3,1,1)
gold = data[data.Medal == "Gold"].Team.value_counts().head(5)
gold.plot(kind='bar',rot=0,figsize=(20, 10))
plt.ylabel("Gold Medal")
plt.subplot(3,1,2)
silver = data[data.Medal == "Silver"].Team.value_counts().head(5)
silver.plot(kind='bar',rot=0,figsize=(20, 10))
plt.ylabel("Silver Medal")
plt.subplot(3,1,3)
bronze = data[data.Medal == "Bronze"].Team.value_counts().head(5)
bronze.plot(kind='bar',rot=0,figsize=(20, 10))
plt.ylabel("Bronze Medal")

plt.show()


# # Assessing India's performance at the Olympic Games      

# In[ ]:


import seaborn as sns
medal=data[data.Medal.notnull()]
Indian_medals=medal[medal.Team=='India']
Indian_medals.head()
sns.countplot(x='Medal',data=Indian_medals)


# # Getting details of the games where India has won medals

# In[ ]:


gold = data[(data.Medal == 'Gold')]
goldIND = gold.loc[gold['NOC'] == 'IND']
goldIND.Event.value_counts().reset_index(name='Medal').head(20)


# # Performance of Indian Hockey team during Dhyan Chand's playing days

# In[ ]:


img=np.array(Image.open('../input/dhyanchand/1936_Olympic_Hockey_Team_1.jpg'))
fig=plt.figure(figsize=(50,10))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.show()


# Dhyan Chand was instrumental in India wining Hockey Olympic gold medals.Watch the video to know few facts about Dhyan Chand.

# 

# In[ ]:


from IPython.display import YouTubeVideo

YouTubeVideo('ImqtglipOz8', width=800, height=450)


# In[ ]:




