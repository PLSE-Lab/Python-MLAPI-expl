#!/usr/bin/env python
# coding: utf-8

# Hello friends, in this kernel I would like to find insights of some features. So, let's get started.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


marvel = pd.read_csv("../input/marvel-wikia-data.csv")
dc = pd.read_csv("../input/dc-wikia-data.csv")

# Any results you write to the current directory are saved as output.


# In[2]:


print ("marvel\n",marvel.sample(3))
print ("DC\n",dc.sample(3))


# As you see, Dataset contains
# * Appearances of Characters, Name of Character, Gender, ALIGN, EYE, HAIR and some more attributes. 
# * I would like to relate ALIGN and these features.

# In[3]:


print("Marvel:",marvel.shape," DC:",dc.shape)


# * So, DC has less characters than that of Marvel.
# * There are many Marvel,DC Appearanes. But I would like to explore on data which have atleast 100 appearances.
# 

# In[4]:


marvel = marvel[marvel.APPEARANCES >= 100]
dc = dc[dc.APPEARANCES>=100]
print("Marvel:",len(marvel)," DC:",len(dc))


# Let's see the columns

# In[5]:


print("MARVEL:",marvel.columns)
print("DC:",dc.columns)


# So, I would like to remove some columns which are not necessary for my data exploration.

# In[6]:


marvel.drop(columns=['ID','urlslug','GSM','FIRST APPEARANCE','Year'],inplace=True)
dc.drop(columns=['ID','urlslug','GSM','FIRST APPEARANCE','YEAR'],inplace=True)


# Let's see whether there are any null values....

# In[7]:


print("MARVEL\n",marvel.isna().sum())
print("DC\n",dc.isna().sum())


# Let's start exploring,

# In[8]:


print("Marvel\n",marvel.ALIGN.value_counts(dropna=False))
print("DC\n",dc.ALIGN.value_counts(dropna=False))


# I would like to replace the missing data of DC,Marvel "ALIGN" attribute to Neutral Characters.

# In[9]:


marvel.ALIGN.fillna(value = "Neutral Characters",inplace = True)
dc.ALIGN.fillna(value = "Neutral Characters",inplace = True)


# Let's check it

# In[10]:


print("MARVEL\n",marvel.ALIGN.isna().sum())
print("DC\n",dc.ALIGN.isna().sum())


# Let's explore on "EYE" column.
# 

# In[11]:


print("MARVEL\n",marvel.EYE.value_counts(dropna=False))
print("DC\n",dc.EYE.value_counts(dropna=False))


# It has many values. So I would like to generalize them.

# In[12]:


eyes = ['Blue Eyes','Brown Eyes','Green Eyes','Red Eyes','Black Eyes']
eyes_after_marvel = []
for i in marvel.EYE.values:
    if i not in eyes:
        eyes_after_marvel.append('Different Eyes')
    else:
        eyes_after_marvel.append(i)
marvel['EYE'] = eyes_after_marvel
eyes_after_dc = []
for i in dc.EYE.values:
    if i not in eyes:
        eyes_after_dc.append('Different Eyes')
    else:
        eyes_after_dc.append(i)
dc['EYE'] = eyes_after_dc
print("MARVEL\n",marvel.EYE.value_counts(dropna=False))
print("DC\n",dc.EYE.value_counts(dropna=False))


# Now, I would like explore data only on Male,Female Characters

# In[13]:


print("Marvel\n",marvel.SEX.value_counts(dropna=False))
print("DC\n",dc.SEX.value_counts(dropna=False))


# In[14]:


#let's take only Male,Female Characters Data
marvel = marvel[marvel.SEX.isin(["Male Characters","Female Characters"])]
dc = dc[dc.SEX.isin(["Male Characters","Female Characters"])]


# So, we left with Hair,
# Since there are many types in Hair, I would like to generalize it.

# In[15]:


print("Marvel\n",marvel.HAIR.value_counts(dropna=False))
print("DC\n",dc.HAIR.value_counts(dropna=False))


# In[16]:


hair = ["Black Hair","Brown Hair","Blond Hair","Red Hair","Bald","No Hair","White Hair","Strawberry Blond Hair","Grey Hair","Auburn Hair"]
hair_after_marvel = []
for i in marvel.HAIR.values:
    if i not in hair:
        hair_after_marvel.append('Different Hair')
    else:
        hair_after_marvel.append(i)
marvel['HAIR'] = hair_after_marvel
hair_after_dc = []
for i in dc.HAIR.values:
    if i not in hair:
        hair_after_dc.append('Different Hair')
    else:
        hair_after_dc.append(i)
dc['HAIR'] = hair_after_dc
print("MARVEL\n",marvel.HAIR.value_counts(dropna=False))
print("DC\n",dc.HAIR.value_counts(dropna=False))


# Let's see whether the updated data has any missing info

# In[17]:


#so now, data doesn't have any missing values
print("MARVEL\n",marvel.isna().sum())
print("DC\n",dc.isna().sum())


# * **Data Exploration**

# In[18]:


print("GOOD Characters of MARVEL EYE color\n",marvel[marvel.ALIGN.isin(['Good Characters'])].EYE.value_counts())
print("GOOD Characters of DC EYE color\n",dc[dc.ALIGN.isin(['Good Characters'])].EYE.value_counts())


# Let's see the percentage of Good and bad characters

# In[19]:


print("MARVEL\n",marvel.ALIGN.value_counts(normalize=True))
print("DC\n",dc.ALIGN.value_counts(normalize=True))
dc_vc = dc.ALIGN.value_counts(normalize=True).reset_index()
marvel_vc = marvel.ALIGN.value_counts(normalize=True).reset_index()
fig, axs = plt.subplots(nrows=2)
plt.subplots_adjust(hspace=0.5)
fig.set_size_inches(10, 8)
sns.barplot(x='index',y='ALIGN',data = dc_vc,ax=axs[0]).set_title('DC')
sns.barplot(x='index',y='ALIGN',data = marvel_vc,ax=axs[1]).set_title('MARVEL')


# Now, We can see that,
# * DC has **71.6%** Good characters
# * Marvel has  **64.5%** of Good Characters
# 
# 

# Let's see some plots and try to find some insights.
# * Is there any relation between EYES and Characters?

# In[20]:


plt.figure(figsize=(10,5))
sns.countplot(x="ALIGN", data=marvel,hue = 'EYE').set_title("MARVEL")


# As we see above, we can say that majority of characters have Blue eyes. But we can't say the proportion.
# Bad Characters of MARVEL mostly have Brown Eyes compared to that of Blue eyes.

# In[21]:


plt.figure(figsize=(10,5))
sns.countplot(x="ALIGN", data=dc,hue = 'EYE').set_title("DC")


# Unless we see in terms of Percentage, we can't analyze.

# In[22]:


plt.figure(figsize=(10,5))
sns.countplot(x="ALIGN", data=marvel,hue = 'HAIR').set_title("MARVEL")


# In[23]:


plt.figure(figsize=(10,5))
sns.countplot(x="ALIGN", data=dc,hue = 'HAIR').set_title("DC")


# **MARVEL**

# In[24]:


character_eyes = marvel.groupby(['ALIGN','EYE']).count().name.reset_index()
character_eyes = character_eyes.groupby(['ALIGN','EYE']).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()
character_eyes
plt.figure(figsize=(15,8))
sns.barplot(x="ALIGN",y='name', hue='EYE',data=character_eyes).set_title("MARVEL")


# **MARVEL**<br>
# We can see that,
# * red eyes(also Green eyes, different eyes) constitute more to Bad Characters than that of those contributing to GOOD characters.
# * Blue and brown eyes constitute to around (70-80 %) for non-bad Characters
# * But for Bad Characters, Blue,brown constitute around 60%
# 

# In[25]:


character_eyes = dc.groupby(['ALIGN','EYE']).count().name.reset_index()
character_eyes = character_eyes.groupby(['ALIGN','EYE']).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()
character_eyes
plt.figure(figsize=(15,8))
sns.barplot(x="ALIGN",y='name', hue='EYE',data=character_eyes).set_title("DC")


# **DC**<br>
# As you see,
# * Blue eyes are common for GOOD,Neutal Characters (>50%) but not for BAD characters
# * Green eyes and Red eyes constitute more percentage for BAD characters compared to that of GOOD characters

# So, we can generalize that,
# * if there's any character with Red eyes, it may align to BAD character

# Let's see what can we find based on Hair and Align

# In[26]:


character_hair = marvel.groupby(['ALIGN','HAIR']).count().name.reset_index()
character_hair = character_hair.groupby(['ALIGN','HAIR']).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()
character_hair
plt.figure(figsize=(15,8))
sns.barplot(x="ALIGN",y='name', hue='HAIR',data=character_hair).set_title("MARVEL")


# **MARVEL** <br>
# We can see that,
# * Bald Hair percentage for Bad Characters is more than Good Characters.
# * Blond Hair Percentage is less for Bad Characters compared to that of Good Characters.
# * Also, Different hair has more percentage to Bad Characters compared to that of Good  Characters.
# * Good Characters has less Brown Hair Percentage than that of Bad Characters
# But still, we can't differentiate Good and Bad characters based on Hair,Eyes

# In[27]:


character_hair = dc.groupby(['ALIGN','HAIR']).count().name.reset_index()
character_hair = character_hair.groupby(['ALIGN','HAIR']).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()
character_hair
plt.figure(figsize=(15,8))
sns.barplot(x="ALIGN",y='name', hue='HAIR',data=character_hair).set_title("DC")


# **DC** <br>
# We can see that,
# * Different Hair percentage for Bad Characters is more than Good Characters.
# * **Blond Hair Percentage is less for Bad Characters compared to that of Good Characters.**
# * Good Characters has less Brown Hair Percentage than that of Bad Characters.
# But still, we can't differentiate Good and Bad characters based on Hair,Eyes

# Let's see the Gender of data. 

# In[28]:


character_gender = marvel.groupby(['ALIGN','SEX']).count().name.reset_index()
character_gender = character_gender.groupby(['ALIGN','SEX']).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()
character_gender
plt.figure(figsize=(15,8))
sns.barplot(x="ALIGN",y='name', hue='SEX',data=character_gender).set_title("MARVEL")


# **MARVEL** <br>
# As we can see,
# it's almost like
# * **Good Characters :**
# *         60 - 40 ratio(Male/Female)
# 
# * **Bad Characters**
# *         85 - 15 ratio(Male/Female)

# In[29]:


character_gender = dc.groupby(['ALIGN','SEX']).count().name.reset_index()
character_gender = character_gender.groupby(['ALIGN','SEX']).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()
character_gender
plt.figure(figsize=(15,8))
sns.barplot(x="ALIGN",y='name', hue='SEX',data=character_gender).set_title("CD")


# **DC** <br>
# As we can see,
# it's almost like
# * **Good Characters :**
# *         70 - 30 ratio(Male/Female)
# 
# * **Bad Characters**
# *         80 - 20 ratio(Male/Female)

# Let's see MARVEL's and DC's most Appearances

# In[30]:


print("Top 10 most appearances of MARVEL\n",marvel.sort_values(by='APPEARANCES',ascending=False)[:10][['name','APPEARANCES']])
print("Top 10 most appearances of DC\n",dc.sort_values(by='APPEARANCES',ascending=False)[:11][['name','APPEARANCES']])


# let's see Death of characters percentage.
# **MARVEL**

# In[31]:


character_alive = marvel.groupby(['ALIGN',"ALIVE"]).count().name.reset_index()
character_alive = character_alive.groupby(['ALIGN',"ALIVE"]).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()
character_alive
plt.figure(figsize=(15,8))
sns.barplot(x="ALIGN",y='name', hue="ALIVE",data=character_alive).set_title("MARVEL")


# So,
# * There are more deaths for Bad Characters compared to that of Good Characters.

# In[32]:


character_alive = dc.groupby(['ALIGN',"ALIVE"]).count().name.reset_index()
character_alive = character_alive.groupby(['ALIGN',"ALIVE"]).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()
character_alive
plt.figure(figsize=(15,8))
sns.barplot(x="ALIGN",y='name', hue="ALIVE",data=character_alive).set_title("DC")


# So,
# * There are more deaths for Neutral Characters compared to that of other characters.

# * I'm noobie. I would like to explore data and it's insights.
# * There's much more to update. Please give any suggestions for exploring further.
# Just an observation...

# In[ ]:




