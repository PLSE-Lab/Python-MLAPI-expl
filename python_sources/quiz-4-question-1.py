#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **1. Import .csv file and display first 5 rows.**
# 
# This shows the first 5 rows of the dataset. 

# In[ ]:


players = pd.read_csv('/kaggle/input/fifa19/data.csv')
players.head()


# 2. This displays the number of rows and columns in the dataset. 
# 
# There are 18,207 rows and 89 columns of data within this set. 

# In[ ]:


players.shape


# **3. Top Player Overall**
# 
# The dataset is ordered by the overall rating of each player. This displays that the player with the highest overall rating is L. Messi (Lionel Messi), as well as his statistics from each column.  

# In[ ]:


players.iloc[0]


# **4. AGE**

# 4a. Soccer players usually retire before they are 40 years old, however, that isn't always the case. This displays all players who are 40 years old or older. 

# In[ ]:


players.query('Age >= 40')


# 4b. This displays the average age of all the players in the dataset. Given that 22 individuals are 40 years old or older, this means that there are players who are significantly younger and that the majority are around 25 years old. 

# In[ ]:


players.Age.mean()


# 4c. This displays the range of ages in the dataset as well as the number of players with the correlating age. Given that the average age is around 25 years old and that 22 players are 40 years old or older, this displays the range of ages and shows how uncommon older individuals are in FIFA. 

# In[ ]:


players.groupby('Age').size().plot.barh()


# **5. Nationalities**

# 5a. This displays the top 10 Nationalities with the most players in the dataset. This dataset is based off of the Club teams that each individual plays for, therefore, certain nationalities have a higher representation. This displays the nationalities with the highest representations within FIFA. 

# In[ ]:


players.groupby('Nationality').size().sort_values(ascending=False).head(10)


# 5b. This displays the Nationality with the lowest representation in FIFA, as well as the player from that country. Rwanda only has one player in the league, D. Bizimana. 

# In[ ]:


min_country = players.groupby('Nationality').size().sort_values(ascending=False).idxmin()
players[ players.Nationality == min_country ]


