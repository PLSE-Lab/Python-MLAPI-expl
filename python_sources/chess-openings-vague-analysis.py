#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv('../input/games.csv')
data.head()


# In[3]:


# Remove variants
def main_openinig_stripper(opening):
    if ':' in opening:
        opening = opening.split(':')[0]
    while '|' in opening:
        opening = opening.split('|')[0]
    if '#' in opening:
        opening = opening.split('#')[0]
    if 'Accepted' in opening:
        opening = opening.replace('Accepted', '')
    if 'Declined' in opening:
        opening = opening.replace('Declined', '')
    if 'Refused' in opening:
        opening = opening.replace('Refused', '')
    return opening.strip()
data['main_opening'] = data.opening_name.apply(main_openinig_stripper)


# In[41]:


# Which is the most played opening?
top_x = 15

plt.figure(figsize=(15,25))
plt.subplot(211)
chart = data.groupby('main_opening').size().nlargest(top_x).plot('bar')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

sizes = np.append(data.main_opening.value_counts().iloc[:top_x].values, data.main_opening.value_counts().iloc[top_x+1:].values.sum())
labels = np.append(data.main_opening.value_counts().iloc[:top_x].index, 'Other')
plt.subplot(212)
plt.pie(sizes, labels=labels, autopct='%.1f%%',
        shadow=True, pctdistance=0.85, labeldistance=1.05, startangle=90, explode = [0 if i > 0 else 0.2 for i in range(len(sizes))])
plt.axis('equal')
plt.show()


# In[5]:


# Drop rare openings that we do not have enough data for
rate = int(len(data) * 0.005) # if played less than 0.1%
data = data.groupby('main_opening').filter(lambda x: len(x) > rate)
# plt.figure(figsize=(15,10))
# chart = data.groupby('main_opening').size().nsmallest(top_x).plot('bar')
# chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
# plt.show()


# The graphs above ignores the idea that a certain player might have played many games, and he probably have a favourite opening that he uses most of the times.
# In order to get which opening is the most famous among all players without giving weigth to the number of games played for players, we will need to use a different approach.

# In[6]:


# Who is the player who played the most games?

top_x = 15
plt.figure(figsize=(15,8))
chart = sns.countplot(data.black_id.append(data.white_id), order=data.black_id.append(data.white_id).value_counts().iloc[:top_x].index)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.show()


# Although this sounds like we can extract some trends here, there are so many other factors that could affect the end result other than the opening.
# Obviously, the skill level of the player is very important. I will try to limit the values to only show winnings that could be interesting for us.
# Something like white winning although black has a higher rating could be more interesting..

# In[7]:


surprise = data[((data.white_rating > data.black_rating) & ((data.winner=='black') | (data.winner == 'draw'))) |
                ((data.white_rating < data.black_rating) & ((data.winner=='white') | (data.winner == 'draw')))]
openings_grouped_counts = surprise.groupby(['main_opening', 'winner']).id.count()
openings_grouped_percentage = openings_grouped_counts.groupby(level=[0]).apply(lambda g: g / g.sum())
openings_grouped = pd.concat([openings_grouped_counts, openings_grouped_percentage], axis=1, keys=['counts', 'percentage'])


# In[38]:


openings_grouped.head(6)


# In[27]:


#let's see strong trends
interesting_openings = openings_grouped[openings_grouped.percentage > 0.5]

plt.figure(figsize=(15,8))
chart = interesting_openings.percentage.nlargest(len(interesting_openings)).plot('bar')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.show()


# So, The chart above describes which openings have some statistical advantage for every color. Note that we have only included 'surprise' winnings where the weaker player won the stronger player.
# Of course, that rating could not reflect the real skill level of the player as he might just be a player who just signed up on the platform or any other reason! Anyways, this was designed for fun and some pandas manipulations.
# And in case you are interested, I will show  you the top opening for every color.

# **For White - Mieses Opening:** ![](https://www.chessvideos.tv/chess-opening-database/static/d3.gif)
# **For Black - Owen Defense:** ![](https://images.chesscomfiles.com/uploads/v1/blog/114072.0fbe2100.5000x5000o.75b13930c140.gif)
