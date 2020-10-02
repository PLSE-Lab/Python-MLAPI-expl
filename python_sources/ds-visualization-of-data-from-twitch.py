#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


#Data is scrubbed from previous processing in SQL.

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd






# Bar Graph: Featured Games

games = ["LoL", "Dota 2", "CS:GO", "DayZ", "HOS", "Isaac", "Shows", "Hearth", "WoT", "Agar.io"]

viewers =  [1070, 472, 302, 239, 210, 171, 170, 90, 86, 71]

plt.figure(figsize=(20,10))
plt.bar(range(len(games)), viewers, color='pink')
plt.title('featured game viewers')
plt.xlabel('games')
plt.ylabel('viewers')
ax=plt.subplot()
ax.set_xticks(range(len(games)))
ax.set_xticklabels(games, rotation = 45)
plt.legend(['Twitch'])
plt.show()


# In[ ]:


# Pie Chart: League of Legends Viewers' Whereabouts

labels = ["US", "DE", "CA", "N/A", "GB", "TR", "BR", "DK", "PL", "BE", "NL", "Others"]

colors = ['lightskyblue', 'gold', 'lightcoral', 'gainsboro', 'royalblue', 'lightpink', 'darkseagreen', 'sienna', 'khaki', 'gold', 'violet', 'yellowgreen']

countries = [447, 66, 64, 49, 45, 28, 25, 20, 19, 17, 17, 279]
plt.figure(figsize=(20,10))
plt.pie(countries, labels = labels, colors = colors, autopct = '%1d%%')
plt.axis('equal')
plt.show()



# In[ ]:


plt.figure(figsize=(20,10))
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
plt.pie(countries, explode=explode, colors=colors, shadow=True, startangle=345, autopct='%1.0f%%', pctdistance=1.15)
#plt.axis('equal')
plt.title("League of Legends Viewers' Whereabouts")
plt.legend(labels, loc="right")
plt.show()


# In[ ]:


# Line Graph: Time Series Analysis

hour = range(24)

viewers_hour = [30, 17, 34, 29, 19, 14, 3, 2, 4, 9, 5, 48, 62, 58, 40, 51, 69, 55, 76, 81, 102, 120, 71, 63]

y_upper = [i + 0.15*i for i in viewers_hour]
y_lower = [i - 0.15*i for i in viewers_hour]

plt.figure(figsize=(20,10))
plt.plot(hour, viewers_hour)
plt.fill_between(hour, y_lower, y_upper, alpha = 0.2)
plt.xlabel("Hour")
plt.ylabel("Viewers")
plt.title("Time Series")
plt.legend(['2015-01-01'])
plt.show()

