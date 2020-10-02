#!/usr/bin/env python
# coding: utf-8

# 
# 
# ## Part - 1 Data Manipulation

# ** Import pandas as pd.**

# In[ ]:


import pandas as pd


# ** Read games.csv as a dataframe called games.**

# In[ ]:


games = pd.read_csv('../input/games-data/games.csv')


# ** Check the head of the DataFrame. **

# In[ ]:


games.head()


# ** Use .info() method to find out total number of entries in dataset**

# In[ ]:


games.info()


# **What is the mean playin time for all games put together ?**

# In[ ]:


games['playingtime'].mean()


# ** What is the highest number of comments received for a game? **

# In[ ]:


games['total_comments'].max()


# ** What is the name of the game with id 1500?  **

# In[ ]:


games[games['id']==1500]['name']


# ** And which year was it published? **

# In[ ]:


games[games['id']==1500]['yearpublished']


# ** Which game has received highest number of comments? **

# In[ ]:


games[games['total_comments']== games['total_comments'].max()]


# ** Which games have received least number of comments? **

# In[ ]:


games[games['total_comments']== games['total_comments'].min()]


# 
# ** What was the average minage of all games per game "type"? (boardgame & boardgameexpansion)**

# In[ ]:


games.groupby('type').mean()['minage']


# ** How many unique games are there in the dataset? **

# In[ ]:


games['id'].nunique()


# ** How many boardgames and boardgameexpansions are there in the dataset?  **

# In[ ]:


games['type'].value_counts()


# ** Is there a correlation between playing time and total comments for the games? - Use the .corr() function **

# In[ ]:


games[['playingtime','total_comments']].corr() # No correlation.


# ## Part 2 
# 
# ## Data Visualization using Seaborn

# ### Import the seaborn library and set color codes as true

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


# ### Drop na values for negating issues during visualization

# In[ ]:


games.dropna(inplace=True)


# ### View the distance plot for minage

# In[ ]:


sns.distplot(games["average_rating"])


# ### Is there a linear relationship between Minage & average_rating?

# In[ ]:


sns.jointplot(games["minage"], games["average_rating"])


# ### Compare the relationship between playingtime , minage and average rating using pairplot

# In[ ]:


sns.pairplot(games[["playingtime", "minage", "average_rating"]])


# ### Compare type of game and playingtime using a stripplot

# In[ ]:


sns.stripplot( games["type"], games["playingtime"], jitter= True)


# ### Analyze the linear trend between playing time(less than 500 mins) and average_rating received for the same

# In[ ]:


sns.lmplot(x= "playingtime", y= "average_rating", data = games[games["playingtime"]<500])

