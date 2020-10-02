#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('../input/chess/games.csv')
df.head()


# In[ ]:


# rank groupings---> find most common opening by rank grouping
# most common opening move 


# Most games seem to take about 50 turns 

# In[ ]:


plt.style.use('dark_background')
sns.distplot(df.turns,color='white')
plt.show()


# Games are typically evenly matched. Most games occur around 1500 elo but are positvely skewed. This is likely because players who tend to play alot of matches could be higher skilled players.

# In[ ]:


plt.style.use('default')
sns.jointplot(x='white_rating',y='black_rating',data=df,kind='hex',color='black')


# Draws are longer games. This makes sense as draws will often drag out the game with little chance of mating their opponent.

# In[ ]:


plt.style.use('default')
colors = ["white", "darkgrey","red"]
customPalette = sns.set_palette(sns.color_palette(colors))
sns.violinplot(x="winner", y="turns", data=df,palette=customPalette)


# Higher Rated games seem to have weak correlation with having more turns. Perhaps higher rated players are less likely to make game altering blunders. Maybe lower rated players get beat quickly by players who are sandbagging from higher ranks. It could also be the case that higher rated players take the game more seriosuly and will draw out games longer in attempts to win.

# In[ ]:


plt.style.use('grayscale')

sns.regplot(x='black_rating',y='turns',data=df,scatter_kws={'s':2})


# In[ ]:


# rated games vs non rated games length
plt.style.use('default')
sns.countplot(df.rated)


# In[ ]:


plt.style.use('default')
sns.countplot(df.victory_status,order=df.victory_status.value_counts().iloc[:4].index)


# Rated games seem to result in more turns. This adds to the theory that people are taking more turns because they are trying harder.

# In[ ]:


my_colors=['red','blue']
df.groupby(['rated']).mean().turns.plot(kind='bar',color=my_colors)


# In[ ]:


sns.countplot(y="black_id", data=df, palette="Reds",
              order=df.black_id.value_counts().iloc[:10].index)


# In[ ]:


plt.style.use('default')
plt.figure(figsize=(30, 10))
sns.countplot(df.opening_name,order=df.opening_name.value_counts().iloc[:8].index)


# In[ ]:




