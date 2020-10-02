#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import matplotlib.dates as mdates
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Purpose
# 
# I like playing around with UFC datasets.  This seems like a wonderful dataset to either use by itself or to fold into other datasets.  I just want to poke around and see what kind of questions this dataset can offer.

# # 1. Load and Clean Data

# In[ ]:


df = pd.read_csv('/kaggle/input/ufc-rankings/rankings_history.csv')


# In[ ]:


df.info(verbose=True)


# We should probably covert the date column and clean any nulls if they exist

# In[ ]:


df['date'] = pd.to_datetime(df['date'])

df = df.dropna()


# In[ ]:


df.info(verbose=True)


# # 2. Let's Answer Some Questions

# ## Who has spent the most time in the rankings?

# In[ ]:


top_fighter_list = (df['fighter'].value_counts().head(10))

display(top_fighter_list)


# In[ ]:


date_list = df.date.unique()


# In[ ]:


print(f"There have been {len(date_list)} rankings released by the UFC.")


# Hmmmm.... There have been only 229 rankings released.  How have the top fighters been ranked much more than this?  There are two possibilities.  First let's take a look at all of the possible lists where a fighter can be ranked:

# In[ ]:


weightclass_list = df.weightclass.unique()
print(weightclass_list)


# As you can see there is a 'pound-for-pound' weight-class.  To quote Wikipedia: 
# 
# Pound for pound is a ranking used in combat sports, such as boxing wrestling, or mixed martial arts, of who the better fighters are relative to their weight (i.e., adjusted to compensate for weight class). As these fighters do not compete directly, judging the best fighter pound for pound is subjective, and ratings vary. They may be based on a range of criteria including "quality of opposition", factors such as how exciting the fighter is or how famous they are, or be an attempt to determine who would win if all those ranked were the same size.
# 
# https://en.wikipedia.org/wiki/Pound_for_pound

# A second possibility is that a fighter could be ranked, or even a champion (champ-champ) in two or more weight classes.  To make a more straightforward comparision of who has been on the rankings lists the longests I am going to go through the data weight-class by weight-class and see who has been ranked the most often in each weight class (including pound-for-pound).

# In[ ]:


top_fighter_lists = []

for w in weightclass_list:
    mask = df['weightclass'] == w
    top_fighter_lists.append(df['fighter'][mask].value_counts().head(10))



        
#print(list_max, list_min)


# Let's take a quick look at this data

# In[ ]:


list_max = 0
list_min = 1000

for z in range(len(top_fighter_lists)):
    #print(z)
    temp_max = max(top_fighter_lists[z])
    if temp_max > list_max:
        list_max = temp_max
        
    temp_min = min(top_fighter_lists[z])
    if temp_min < list_min:
        list_min = temp_min


# In[ ]:


for z in range(len(top_fighter_lists)):
    temp_df = pd.DataFrame(top_fighter_lists[z])
    temp_df.columns = [weightclass_list[z]]
    #display(temp_df)
    fig, ax = plt.subplots(figsize=(3,7))
    sns.heatmap(temp_df, annot=True, fmt=".4g", cmap='Reds', ax=ax, vmin=list_min, vmax=list_max)
    plt.yticks(rotation=0, fontsize=12)
    plt.title(weightclass_list[z], fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)


# ## What does the journey of these ranking mainstays look like?

# In[ ]:





for z in range(len(weightclass_list)):
    wc = weightclass_list[z]
    f = top_fighter_lists[z].keys()[0]

    rank_list = []

    for d in date_list:
        #print(d)
        temp_df = df[(df['date']==d) & (df['fighter'] == f) & (df['weightclass'] == wc)]
        if (len(temp_df)) > 0:
            rank_list.append((temp_df['rank']).values[0])
        else:
            rank_list.append(np.nan)
    #print(f"{f} at {wc} rankings: ")
    #display(rank_list)
    
    rank_list = np.asarray(rank_list)
    lower = 0
    slower = np.ma.masked_where(rank_list < .5, rank_list)    
    fig, plt.figure(figsize=(9,5))
    
    plt.plot(date_list,  rank_list, date_list, slower)

    plt.gca().invert_yaxis()
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Rank', fontsize=16)
    plt.title(f'{f} {wc} Rank', fontweight='bold', fontsize=16)
    plt.show()


# # What does rank movement in a single weight class look like?

# In[ ]:


for z in range(len(weightclass_list)):
    wc = weightclass_list[z]

    if len(top_fighter_lists[z]) > 3:
        fighters = [top_fighter_lists[z].keys()[0], top_fighter_lists[z].keys()[1], top_fighter_lists[z].keys()[2], 
               top_fighter_lists[z].keys()[3]]
    else:
        fighters = [top_fighter_lists[z].keys()[0], top_fighter_lists[z].keys()[1], top_fighter_lists[z].keys()[2]]

        
    rank_list_list = []
    
    for f in fighters:
        rank_list = []
        for d in date_list:
            temp_df = df[(df['date']==d) & (df['fighter'] == f) & (df['weightclass'] == wc)]
            if (len(temp_df)) > 0:
                rank_list.append((temp_df['rank']).values[0])
            else:
                rank_list.append(np.nan)
        #display(rank_list)
        rank_list_list.append(rank_list)
    #print(f"Fighters: {fighters}")
    #display(rank_list_list)
    fig, plt.figure(figsize=(9,5))
    if len(top_fighter_lists[z]) > 3:                    
        plt.plot(date_list,  rank_list_list[0], date_list,  rank_list_list[1], date_list,  rank_list_list[2],
             date_list,  rank_list_list[3])
    else:
        plt.plot(date_list,  rank_list_list[0], date_list,  rank_list_list[1], date_list,  rank_list_list[2])
                    
    plt.gca().invert_yaxis()
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Rank', fontsize=16)
    plt.title(f'{wc} Rank', fontweight='bold', fontsize=16)
    plt.legend(fighters)
    plt.show()    


# In[ ]:




