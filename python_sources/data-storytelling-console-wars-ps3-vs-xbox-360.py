#!/usr/bin/env python
# coding: utf-8

# ## 1. Introduction
# ---
# For the longest time, my friends and I debated which console was better- PS3 or Xbox 360. These long late night debates have raged on from middle school and I want to put an end to them once and for all. From arguing about the console specs, to the games, to the feel of the controller, to any other insignificant detail- we have done it all.
# 
# Having grown up, at least in the field of computer science, I wanted to take an analytical approach. Forget the "emotional battles", lets look at the raw stats and see which console performed better.
# 
# I want to close this chapter of my life before heading out into the real world. Let's see what the results say...
# 
# #### Note: This analysis is done using sales alone.

# In[9]:


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


# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt

# changing the font size in sns
sns.set(font_scale=3)


# In[11]:


df = pd.read_csv('../input/vgsales.csv', encoding='utf-8')
df.head()


# In[13]:


# counts the number of items per Platform
sns.factorplot('Platform', data=df, kind='count', size=10, aspect=2)

# aesthetics
plt.title('Initial Plot')
plt.xlabel('Platform')
plt.ylabel('Number of Games Released')
plt.xticks(rotation=90)

# display
plt.show()


# In[15]:


gen2 = ['Wii', 'X360', 'PS3']
df_g2 = df[df['Platform'].isin(gen2)]

sns.factorplot('Platform', data=df_g2, kind='count', size=10, aspect=2)

plt.title('Number of Unique Games Released for Gen. 2')
plt.xlabel('Platform')
plt.ylabel('Number of Games')
plt.xticks(rotation=90)

plt.show()


# In[16]:


sns.factorplot('Year', hue='Platform', data=df_g2, kind='count', size=10, aspect=2)

plt.title('Number of Games Released for Year in Gen. 2')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.xticks(rotation=90)

plt.show()


# ### Analysis
# ---
# Each console in generation 2 has a similar number of unique games released. Let's look at the **game sales** to have a better understanding of which console stands out!

# In[18]:


df_g2_sales = df_g2.groupby(['Platform', 'Year'], as_index=False)[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].sum()
df_g2_sales.head()


# In[21]:


palette ={"PS3":"Blue","X360":"Green", "Wii":"Red"}

sns.factorplot(x='Year', y='Global_Sales', hue='Platform', 
                   data=df_g2_sales, kind='bar', size=10, aspect=2, palette = palette)

plt.title('Global for Gen. 2')
plt.xlabel('Year')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=90)

plt.show()


# In[22]:


sns.factorplot(x='Genre', y='Global_Sales', hue='Platform', 
                   data=df_g2, kind='bar', size=10, aspect=2, palette = palette, n_boot = False)

plt.title('Global Sales for Gen. 2')
plt.xlabel('Genre')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=90)

plt.show()


# ### Analysis 
# ---
# Looks like the Xbox 360/PS3 are neck and neck. PS3 started of a little slow in sales (in the global graph) and started to overtake Xbox 360 in the latter years. This is probably because a PS3 COST NEARLY $600 when it first came out! Eventually, the prices evened out and PS3 had free online so maybe that's why it over took Xbox 360 in the latter years!
# 
# But what about the total sales? Which console had the most sales?

# In[23]:


display(df_g2_sales.groupby(['Platform'])[['Global_Sales']].sum())


# ### Analysis
# ---
# **Xbox 360 WINS in total sales (but not by much)! Why does X360 come out on top?** Let's look at the top 10 games of each console to find out.

# In[37]:


# top Xbox 360 ONLY games
df_g2_X360 = df[df['Platform'] == 'X360']['Name'][:50]
df_g2_PS3 = df[df['Platform'] == 'PS3']['Name']
df_g2_X360_only = df_g2[df_g2['Name'].isin(df_g2_X360) & (df_g2['Name'].isin(df_g2_PS3) == False)]

sns.factorplot(x='Name', y='Global_Sales', data=df_g2_X360_only, kind='bar', size=10, aspect=2)

plt.title('Top X360 Only Games')
plt.xlabel('Games')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=90)

plt.show()


# In[38]:


# top PS3 ONLY games
df_g2_X360 = df[df['Platform'] == 'X360']['Name']
df_g2_PS3 = df[df['Platform'] == 'PS3']['Name'][:50]
df_g2_PS3_only = df_g2[df_g2['Name'].isin(df_g2_PS3) & (df_g2['Name'].isin(df_g2_X360) == False)]

sns.factorplot(x='Name', y='Global_Sales', data=df_g2_PS3_only, kind='bar', size=10, aspect=2)

plt.title('Top PS3 Only Games')
plt.xlabel('Games')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=90)

plt.show()


# ### Analysis
# ---
# **It is possible that Xbox 360 came out on top for two reasons (according to the data).**
# 1. It was release a year earlier, so a lot of sales could have gone out then.
# 2. Xbox 360's exclusive games (Halo, Gears of War, etc.) are largely multiplayer games, while PS3's exlusive games (MGS4, Uncharted, etc.) are largely single player games. So if a friend of mine bought an Xbox 360 game, it might be more likely that I buy it as well- since we can play together on Xbox Live.
# 

# ### Thank you for reading!
# ---
# You can get the full story analysis here: https://github.com/shanvith/data_storytelling/tree/master/2_console_wars

# In[ ]:




