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


# Python has a big eco-system for data analysis, this includes numpy, ipython, jupyter as some lower level modules, then followed by scipy, matplotlib, pandas, and on top of these, it has scikit-learn, statsmodels, etc.  In fact, python with scientifc computing really makes a good combination. With so many modules to learn, I would start learning from the basic, and then expand more in python scientific computing.
# 
# Today I just want to use some of the basic functionalities that I learned in using numpy, pandas and matplotlib, to make a complete data analysis.

# In[ ]:


potw = pd.read_csv('../input/NBA_player_of_the_week.csv')
potw.head()


# In[ ]:


potw.tail()


# It can be seen that there exist two formats for player's height, in foot and in centimeters.  Also some data in conference is missing. 

# In[ ]:


potw.describe()


# We need to deal with missing data first. So we need to find out how many columns contain missing data.

# In[ ]:


potw.apply(lambda x: x.count(), axis=0)


# It is clear that only the column 'Conference' contains missing data.

# In[ ]:


potw_no_conference = potw.dropna(how = 'any')


# In[ ]:


potw_no_conference.apply(lambda x : x.count(), axis=0)


# In[ ]:


potw_no_conference['Age'].hist()


# In[ ]:


potw['Age'].hist()


# There is not much difference of the distribution of players' age between the complete dataset and the dataset that contains players' conference information.
# 
# Let's see how the number of players change during draft each year

# In[ ]:


## here we use the groupby function
player_by_year = potw['Player'].groupby(potw['Draft Year'])


# In[ ]:


player_by_year.count()


# In[ ]:


player_by_year.count().plot(title="The Number of Players Won POTW according to Draft Year")


# The number of players that won POTW the most came from draft year 2003, this is the golden generation when Lebrown, Dwyane entered the league.
# Following this idea, we can now tell which team has the players that won POTW the most, which position that won the POTW the most.
# Let's get to business.

# In[ ]:


player_by_team = potw['Player'].groupby(potw['Team'])


# In[ ]:


player_by_team.count().plot.pie(title="POTW by Teams", figsize=(10,10))


# Of all the teams, Los Angeles Lakers obviously won the POTW the most, followed by teams like Miami Heat, San Antonio Spurs and Cleveland Cavaliers.

# In[ ]:


player_by_position = potw['Player'].groupby(potw['Position'])


# In[ ]:


player_by_position.count().plot.pie(title="POTW by Positions", figsize=(10,10))


# Now it comes to our final question, which player won POTW the most. Right now I'm guessing Lebrown James, Hahaha
# 

# In[ ]:


player_by_name = potw['Player'].groupby(potw['Player'])


# In[ ]:


player_by_name.count()


# In[ ]:


player = pd.DataFrame(player_by_name.count())


# In[ ]:


player.sort_values(by='Player', ascending=False)


# It is actually Lebrown James. And he played for  both Cleaveland and Miami and I think this contributed to the fact that both teams  took a huge chunk from the pie chart above.

# In[ ]:




