#!/usr/bin/env python
# coding: utf-8

# # Historical facts about the "Golden Globes Awards"
# 
# The Golden Globes is an award show hosted in the month of January that has a seventy-seven year history of recognising excellence in film and television. Using a database comprising details of all Golden Globe winners in the award's history, I will provide some quick facts about the show's winners and some recognisable patterns in the show's history. 
# 
# <img src="https://www.indiewire.com/wp-content/uploads/2016/12/golden-globes.jpg"  width="300" height="300">

# In[ ]:


import numpy as np 
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as pl


# ## Structure of the Database

# In[ ]:


df = pd.read_csv("../input/golden-globe-awards/golden_globe_awards.csv")
df.head()


# In[ ]:


df.tail()


# In[ ]:


print('The database contains a total of %d rows.' %df['year_film'].count())


# # **Fact-finding Analysis**
# 
# ## *The Winners Table*
# 
# Let's explore some fact about the nominees and films that have broken records in Golden Globe history. 
# 

# In[ ]:


#Narrow down the database to show the winners only
df_wins = df[df['win'] == True]


# In[ ]:


df_wins.head()


# In[ ]:


df_wins.tail()


# In[ ]:


print('This updated "wins" database contains a total of %d rows.' %df_wins['year_film'].count())


# * Which nominee has been awarded the most number of awards in one night

# In[ ]:


grouped = df_wins.groupby('year_award')

big_winner = grouped['nominee'].value_counts()
pd.DataFrame(big_winner[big_winner == big_winner.max()])


# * We can ask the same question for films and TV shows. Which of these managed to sweep the most number of awards during one ceremony?

# In[ ]:


film_most_wins = grouped['film'].value_counts()
pd.DataFrame(film_most_wins[film_most_wins == film_most_wins.max()])


# * Which films came had the second highest number of wins during one ceremony?

# In[ ]:


# With 5 or more wins in one ceremony
film_most_wins = grouped['film'].value_counts()
pd.DataFrame(film_most_wins[film_most_wins >= 5])


# ### How frequent is it that a film wins more than five awards?

# In[ ]:


pl.figure(figsize=(6,5))
ax = pl.gca()
fs = 13
min_wins = 2
max_wins = grouped['film'].value_counts().max()
sns.set(style='darkgrid')
sns.distplot(film_most_wins[film_most_wins >= float(min_wins)], color='purple', 
             kde=0)
pl.xticks(np.arange(min_wins, max_wins+1, 1))
pl.ylabel('Frequency of Wins', fontsize=fs)
pl.xlabel('Golden Globe wins during one ceremony',fontsize=fs)
pl.show()


# ### Of course, being the biggest winner during one ceremony does not exactly translate to being one of the biggest winners in Golden Globe history. 
# 
# * Which film(s) or TV show(s) have won the most Golden Globes overall? 

# In[ ]:


film_most_wins_history = df_wins.groupby('film')['win'].sum()
pd.DataFrame(film_most_wins_history[film_most_wins_history >= film_most_wins_history.max()])


# In[ ]:


film_most_wins_history = df_wins.groupby('film')['win'].sum()
min_wins_history = 3
max_wins_history = int(film_most_wins_history.max())
sns.set(style='darkgrid')
sns.distplot(film_most_wins_history[film_most_wins_history >= float(min_wins_history)], color='r', 
             kde=0)
pl.xticks(np.arange(min_wins_history, max_wins_history+1, 1))
pl.ylabel('Frequency of Wins', fontsize=fs)
pl.xlabel('Golden Globe wins in history', fontsize=fs)
pl.show()


# *  Which nominee has had the most wins throughout the history of this award show? 

# In[ ]:


name_most_wins = df_wins['nominee'].value_counts().idxmax()
most_wins_count = df_wins['nominee'].value_counts().max()
print('%s has received %d Golden Globe wins in total.' %(name_most_wins,most_wins_count) )


# * Who has received the most nominations in history? 

# In[ ]:


name_most_noms = df['nominee'].value_counts()
print('%s has received %d Golden Globe nominations in total.' %(name_most_noms.idxmax(),name_most_noms.max()) )


# * Which nominees have also received a high number of Golden Globe nominations?

# In[ ]:


pd.DataFrame(df['nominee'].value_counts().head(10))


# * Which film/TV show has received the most nominations in history?

# In[ ]:


film_most_noms = df['film'].value_counts()
print('"%s" has received %d Golden Globe nominations in total.' %(film_most_noms.idxmax(),film_most_noms.max()) )


# * Which other films/TV shows have been nominated this many times? 

# In[ ]:


pd.DataFrame(df['film'].value_counts().head(10))


# ## *Special Awards*
# 
# ### The Cecil B. DeMille Award is an honorary Golden Globe Award bestowed by the Hollywood Foreign Press Association (HFPA) for "outstanding contributions to the world of entertainment".
# 
# * How many actresses/actors have received this honour?

# In[ ]:


df[df['category'].str.contains('Cecil')]['nominee'].count()


# In[ ]:


pd.DataFrame(df[df['category'].str.contains('Cecil')]['nominee'])


# ### A new Golden Globes honorary award has been introduced recently - the Carol Burnett Award. It is an honorary Golden Globe Award bestowed by the Hollywood Foreign Press Association (HFPA) for "outstanding contributions to television on or off the screen." 
# * How many actresses or actors have bestowed with this honour? 

# In[ ]:


df[df['category'].str.contains('Carol Burnett Award')]['nominee'].count()


# In[ ]:


pd.DataFrame(df[df['category'].str.contains('Carol Burnett Award')]['nominee'])

