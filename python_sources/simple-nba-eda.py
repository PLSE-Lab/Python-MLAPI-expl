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
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


players=pd.read_csv('../input/Players.csv')
Seasons=pd.read_csv('../input/Seasons_Stats.csv')


# In[ ]:


players.head()


# In[ ]:


Seasons.head()


# ### First Lets Start With Players Analysis

# In[ ]:


players.shape #shape of the dataframe


# In[ ]:


players.isnull().sum()    #checking data quality


# We can see that there is a null value in a players name. Thus we should delete it

# ### Players Data Cleaning

# In[ ]:


players.drop('Unnamed: 0' ,axis=1,inplace=True)


# In[ ]:


players.dropna(how='all',inplace=True) #dropping the player whose value is null


# In[ ]:


players.set_index('Player',inplace=True) #setting the player name as the dataframe indexx
players.head(2)


# ### Basic Analysis

# In[ ]:


print('The Tallest Player in NBA History is:',players['height'].idxmax(),' with height=',players['height'].max(),' cm')
print('The Heaviest Player in NBA History is:',players['weight'].idxmax(),' with weight=',players['weight'].max(),' kg')


# In[ ]:


print('The Shortest Player in NBA History is:',players['height'].idxmin(),' with height=',players['height'].min(),' cm')
print('The Lightest Player in NBA History is:',players['weight'].idxmin(),' with weight=',players['weight'].min(),' kg')


# In[ ]:


print('The average height of NBA Players is ',players['height'].mean())
print('The average weight of NBA Players is ',players['weight'].mean())


# The average height comes around 6.5 feet. That's more than the average height of people in many countries..:p

# ### Distribution Of Heights

# In[ ]:


bins=range(150,250,10)
plt.hist(players["height"],bins,histtype="bar",rwidth=1.2,color='#0ff0ff')
plt.xlabel('Height in Cm')
plt.ylabel('Count')
plt.axvline(players["height"].mean(), color='b', linestyle='dashed', linewidth=2)
plt.plot()


# **Observations:**
# The heights of the players are majorly in the range 200-210 cm followed by range 190-200 cm. The mean height as seen is around 199 cm

# ### Distribution Of Weights

# In[ ]:


bins=range(60,180,10)
plt.hist(players["weight"],bins,histtype="bar",rwidth=1.2,color='#4400ff')
plt.xlabel('Weight in Kg')
plt.ylabel('Count')
plt.axvline(players["weight"].mean(), color='black', linestyle='dashed', linewidth=2)
plt.plot()


# **Observations:** 
# The weights of players are majorly in the range 90-100 kgs and the mean weight is around 95 kg

# ### Colleges Giving Maximum Players 

# In[ ]:


college=players.groupby(['collage'])['height'].count().reset_index().sort_values(by='height',ascending=False)[:10]
college.set_index('collage',inplace=True)
college.columns=['Count']
ax=college.plot.bar(width=0.8)
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.35))
plt.show()


# In[ ]:


city=players.groupby(['birth_state'])['height'].count().reset_index().sort_values(by='height',ascending=False)[:10]
city.set_index('birth_state',inplace=True)
city.columns=['Count']
ax=city.plot.bar(width=0.8,color='#ab1abf')
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()


# California has the highest number of players followed by New-York.

# ### Season's Analysis

# In[ ]:


Seasons.drop('Unnamed: 0',axis=1,inplace=True)


# In[ ]:


Seasons.head()


# In[ ]:


Seasons.isnull().sum()


# We can see that 67 rows have no player names. We need to drop these.

# In[ ]:


Seasons=Seasons[Seasons['Player'] !=0] #removing entries without names
players.reset_index(inplace=True)


# In[ ]:





# In[ ]:




