#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn # data visualization
import matplotlib.pyplot as ml # data visualisation as well
import warnings

sn.set(color_codes = True, style="white")
warnings.filterwarnings("ignore")
battle = pd.read_csv("../input/battles.csv", sep=",", header=0)#symbol used to separate the value, header at 0 is the name of each column
death = pd.read_csv("../input/character-deaths.csv", sep=",", header=0)#symbol used to separate the value, header at 0 is the name of each column
prediction = pd.read_csv("../input/character-predictions.csv", sep=",", header=0)#symbol used to separate the value, header at 0 is the name of each column


# In[ ]:


battle.shape


# In[ ]:


death.shape


# ## Let's look at the names of the colums for each data file.

# In[ ]:


battle.columns


# In[ ]:


death.columns


# ## Now let's take a peek at the head and tail of each data file.

# In[ ]:


battle.head()


# In[ ]:


battle.tail()


# In[ ]:


death.head()


# In[ ]:


death.tail()


# ## Taking a look at the basic statistical analysis of the three data sets.

# In[ ]:


battle.describe()


# Columns 'defender_3' and 'defender_4' appear to be irrelevant to the data, so let's drop them from the analysis.

# In[ ]:


battle = battle.drop(['defender_3', 'defender_4'],1)


# In[ ]:


death.describe()


# ## Now let's check to see if there's any correlation in the data sets

# In[ ]:


battle.corr()


# In[ ]:


ml.figure(figsize=(20,10)) 
sn.heatmap(battle.corr(),annot=True)


# For the battle data set, it doesn't look like there's any correlations between any particular points.

# In[ ]:


death.corr()


# In[ ]:


ml.figure(figsize=(20,10)) 
sn.heatmap(death.corr(),annot=True)


# It looks like there's some strong correlation between DwD (Dancing with Dragons) and the Death Year. And then again with DwD and Book of Death. I haven't read the book, but i'm assuming that a large number of deaths occur around this book.

# 

# # Let's take a look at which gender has the most deaths.
# 
# I think the answer to that question will be obvious. 0 represents the person is female and 1 represent the person is male.

# In[ ]:


ml.figure(figsize = (15,10))
sn.countplot(x='Gender', data=death)


# In[ ]:


death[death['Gender']==0]['Gender'].value_counts()


# In[ ]:


death[death['Gender']==1]['Gender'].value_counts()


# Yeouch, there has been 760 male character deaths throughout the Game of Thrones series.
# 
# ## Let's take at the look at the battles then.

# In[ ]:


ml.figure(figsize = (15,10))
sn.countplot(x='attacker_outcome',data = battle)


# Interestingly enough, there hasn't been a lot of losses throughout the book. Then, let's take a look at who's initiated the battles.

# In[ ]:


ml.figure(figsize = (15,10))
attack = pd.DataFrame(battle.groupby("attacker_king").size().sort_values())
attack = attack.rename(columns = {0:'Battle'})
attack.plot(kind='bar')


# Not really much of a surprise there. Now let's see which king has had to frequently defend themselves.

# In[ ]:


ml.figure(figsize = (15,10))
defend = pd.DataFrame(battle.groupby("defender_king").size().sort_values())
defend = defend.rename(columns = {0:'Defense'})
defend.plot(kind='bar')


# Again, this isn't really much of a surprise. I just didn't think it would be Robb Stark who would have to defend himself so much compared to Joffrey/Tommen Baratheon.
# 
# ## Continuing off the king data, let's see the win/loss situation.

# In[ ]:


ml.figure(figsize = (15,10))
sn.countplot(x='attacker_king', hue = 'attacker_outcome', data = battle)


# Intersting that for Joffrey/Tommen Baratheon, they attack the most and have to frequently defend themselves and yet they have a pretty high win ratio.

# In[ ]:


ml.figure(figsize = (15,10))
sn.barplot(x='attacker_outcome', y='defender_size', data=battle, estimator = np.mean)


# Heads up! The bigger your army, the more likely it is your attacker will lose. Surprising no one ever. But now let's see how likely it is for an attacker to win depending on the circumstances.

# In[ ]:


ml.figure(figsize = (15,10))
sn.countplot(x='attacker_outcome', hue= 'battle_type', data = battle)
ml.legend(bbox_to_anchor=(1, 1), loc=2)


# Now just out of curiosity, let's see which year the most battles occur.

# In[ ]:


ml.figure(figsize = (15,10))
sn.countplot(x='year', data = battle)


# I don't have any clue as to what's going on in Year 299, but that is not the year to be in for the Game of Thrones world.
