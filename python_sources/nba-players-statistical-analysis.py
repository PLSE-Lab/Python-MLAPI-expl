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


# 
# 1. Read the data as a pandas dataframe and display the first 5 rows

# In[ ]:


import pandas as pd
all_seasons = pd.read_csv("../input/nba-players-data/all_seasons.csv")
all_seasons.head()


# 
# 2. Display the number of rows and columns in the database.

# In[ ]:


all_seasons.shape


# 
# 3. Display Lebron James' point averages for each of his seasons in the database.

# In[ ]:


all_seasons.loc[all_seasons.player_name == 'LeBron James', ['player_name', 'pts', 'season']]


# 
# 4. List all the rows for Spanish players.

# In[ ]:


all_seasons.query("country == 'Spain'")


# 
# 5. Calculate the average assists per season for Chris Paul
# 

# In[ ]:


all_seasons.loc[all_seasons.player_name == 'Chris Paul'].ast.mean()


# 
# 6. Group the dataset by player then season in ascending order.

# In[ ]:


grouped = all_seasons.groupby(['player_name', 'season'])
grouped.first()


# 
# 7. For each player, determine the number of seasons they have played since 1996.

# In[ ]:


seasons_played = all_seasons.player_name.value_counts()
seasons_played.head()


# 
# 8. Plot Lebron James' average points per season.

# In[ ]:


import matplotlib.pyplot as plt

lebron_statistics = all_seasons.loc[all_seasons.player_name == 'LeBron James', ['pts','season']]
plt.plot(lebron_statistics["season"], lebron_statistics["pts"])
plt.ylabel('Points Per Game')
plt.xticks(rotation=90)
plt.title("Lebron James Average Points Per Season")
plt.xlabel('Season')
plt.show()


# 
# 9. Sort the dataset descending by rebounds per game and display the top 10 single season performances.

# In[ ]:


all_seasons['pts'].sort_values(ascending=False).head(10)


# 
# 10. What is the highest points per game that Kobe Bryant achieved in a single season?

# In[ ]:


all_seasons2 = all_seasons.loc[all_seasons.player_name == 'Kobe Bryant']
all_seasons2['pts'].max()

