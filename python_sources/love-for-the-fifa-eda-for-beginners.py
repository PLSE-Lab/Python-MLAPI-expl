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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


df = pd.read_csv('../input/data.csv')

#looking at the data
print(df.head(10))


# In[ ]:


#let's see the type of columns that we have in our data
print(df.columns)
print('TOTAL NUMBER OF COLUMNS : {}'.format(len(df.columns)))


# It can be seen that we are given with columns relating to different information about our players. For instance the nationality of the players, the various position the the palyers play in, their different attributes showing their dribbling ability, positioning, agility, balll control etc. So we do have quite a bit of data on our hands so it becomes quite interesting to analyze all the players according to their given attributes. 

# In[ ]:


print(df.info())


# In[ ]:


sns.set(style ="dark", palette="colorblind", color_codes=True)
plt.figure(figsize = (16,10))
sns.countplot(df['Preferred Foot'])
plt.title('COUNT OF PREFERRED FOOT')


# **Well well well, what do we have here, most of the players are right footed (as expected). But the amount of miss balance is too high. We can easily say that most of the players in the football world preffer their right foot than their left.**

# In[ ]:


sns.set(style ="dark", palette="colorblind", color_codes=True)
plt.figure(figsize = (16,12))
sns.violinplot(y = df['SprintSpeed'])
plt.title('SPRINT SPEED DISTRIBUTION')


# In[ ]:


print(max(df['SprintSpeed'].values))
print(min(df['SprintSpeed'].values))

sns.set(style ="dark", palette="colorblind", color_codes=True)
plt.figure(figsize = (20, 16))
sns.countplot(y = df['SprintSpeed'].values[:100])
plt.ylabel('SPRINT SPEEDS', fontsize = 16)
plt.title('SPRINT SPEEDS OF THE PLAYERS', fontsize = 20)


# **Considering the first 100 players, the maximum sprint speed for a player comes out to be 96 (would be intersting to see who that player is ) and the mimimum sprint speed for a player turns out to be 43. Most of the first 100 players seem to have sprint speed of aroung 75. To be honest 75 as the most common sprint speed for the players is pretty good (not too slow and not too fast). But it would be interesting to see the top players having the highest sprint speed.**

# In[ ]:


df1 = df['Nationality'].head()
df2 = df['Value'].head()
df3 = df['Name'].head()
conc_data = pd.concat([df1,df2,df3],axis =1) # axis = 0 : adds dataframes in row
print(conc_data)


# In[ ]:


df['Potential'].plot(kind = 'line', color = 'g', label = 'Reactions', linewidth = 1, alpha = 0.5,grid = True,linestyle = ':')
df['Overall'].plot(color = 'r',label = 'Overall', linewidth = 1, alpha = 0.5, grid = True, linestyle = '--')


# In[ ]:


# Histogram: number of players's age
sns.set(style ="dark", palette="colorblind", color_codes=True)
plt.figure(figsize=(16,8))
sns.distplot(df.Age, bins = 58, kde = False, color='r')
plt.xlabel("Player\'s age", fontsize=16)
plt.ylabel('Number of players', fontsize=16)
plt.title('Histogram of players age', fontsize=20)
plt.show()


# **Well as expected most of the players lie in the the age group of 20-26 years. You can still see that there are some players from 16 to 18 yeras of age that play at the highest positions but such players are very less. Most of the players below 16 are still in the acedemy or in the youth club trying to  break through into the first team. But as it can be seen players lying between 2- to 26 yeras are the ones that play at the top most level as they have the required experiance to play such a top level. Also after 30 the players are on the verge of retiring as can be seen from the decrasing trend of the above plot. Once a player cross the 30 mark, they start to think about their retirement but ther are few players who still play untill they touch 35. But once the 35 mark is crossed players usually retire. From the above plot you can still find some players of age 40 who play at the top level (which are usually very rare).**

# In[ ]:


# Compare six clubs in relation to age
club_names = ('Real Madrid', 'Liverpool', 'Juventus', 'Manchester United', 'FC Barcelona')
df_club = df.loc[df['Club'].isin(club_names) & df['Age']]

fig, ax = plt.subplots()
fig.set_size_inches(15, 10)
ax = sns.boxplot(x = "Club", y = "Age", data = df_club);
ax.set_title('Distribution of age in some clubs', fontsize=20);


# **Form the above boxplot there are few things that we can conclude.**
# 
# **1. From the lot of some top European clubs that we have considered, Juventus has the oldest team squad. Their overall average age of the squad is around 28 years with the player of age 32 years being the oldest. The Juventus squad mostly consists of older players.**
# 
# **2. On the contrary Real Madrid and Liverpool has some of the most young sqaud in the Europe. For Real Madrid the average age of the sqaud is about 21 years which is way more younger than Juventus. This shows that REAL has more inclination towards young talent in the game. This shows Florentino Perez is more inclined towards building the suaqd considering the future of the club. Same analysis can be done for Liverpool as well. Even though Liverpool is not as young player centric as REAL but the avergae age of the squad for the LIVERPOOL team is around same as that for REAL.**
# 
# **3. Following REAL, BARCA have the most youngest squad. Barcelona has good balance of youth and experiance in their squad. MANU on the other hand has the squad average of 25 years. The miniumum age of the player for the Juventus squad is equivalent to the average age of the MANCHESTER UNITED squad. From this anaylsis you can get the idea of the age difference of the Juventus players as compared to layers of the other teams.**

# In[ ]:


# let's see the total number of players that we have at different playing positions
plt.figure(figsize = (16, 10))
sns.countplot(y = df['Position'], data = df, palette = 'plasma');
plt.title('Count of players on the position', fontsize=20);


# **The above countplot gives us the idea about the number of players that we have at various palying positions.**
# 
# **1. The highest number of players we find play as Strikers. So most of the players that we have in the game are strikers. Well this isn't shocking considering the amount of strikers that we see in all the teams.**
# 
# **2. Following strikers, Goalkeepers are amount to second highest. Most of the teams have at least tow GK's in their squad.**
# 
# **3. On the third we find Centre Backs.**
# 
# **4. Centre Midfielders, Left and Right wingers are also an important part of any squad and thus have good count of players at those positions.**
# 
# **5. There are also certain positions such as RAM, LAM, LF etc that aren't that popular with the management tactics. You won't be able to find large number of players at these postions as these are the least concerned postion or you can say these are not as popular as other postions when it comes to manager tactics.**

# In[ ]:


#let's see wo are the top players with 'LEFT' foot as their preffered foot
data_left = df[df['Preferred Foot'] == 'Left'][['Name', 'Overall', 'Age', 'Potential', 'FKAccuracy']]
print(data_left.head(10))


# In[ ]:




