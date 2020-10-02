#!/usr/bin/env python
# coding: utf-8

# WOW is without a doubt the legendary MMORPG played by millions of people around the world, and to this day, the mention of this game gives me a deep sense of nostalgia. Therefore, having seen this dataset, I had a tireless desire to study it. This is my first kernel on Kaggle, so feel free to comment and critique it.
# 
# So, less words, more action, lets start.

# In[ ]:


# Loading all necessary modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import cufflinks

cufflinks.go_offline(connected=True)
init_notebook_mode(connected = True)
get_ipython().run_line_magic('matplotlib', 'inline')

# To see more columns
pd.set_option('display.max_columns', 100)


# In[ ]:


# loading dataset
data = pd.read_csv('../input/WoW Demographics.csv')
data.head()


# So, we see that our dataset consist of next columns:
# *     Timestamp: Useless. Just when the survey was completed.
# *     Gender: The gender of the player.
# *     Sexuality: The sexuality of the player.
# *     Age: Age of the player.
# *     Country: Country the player lives in.
# *     Main: The gender of character the player mains
# *     Faction: The faction the player mains.
# *     Server: The server(s) the player mains.
# *     Role: The role(s) the player mains (DPS, Healer, Tank, or any combination therein).
# *     Class: The class(es) the player mains. This was a question where the respondent check any number of boxes, so there are many different ways it could be answered. Hard to analyze.
# *     Race: The race(s) the player mains. Same as class.
# *     Max: The number of 110s (max level) the player has. Only numerical variable in the dataset.
# *     Attracted: The gender the player is attracted to.
# *     Type: The "type" of person the player is. Combines gender and sexuality ("gay woman", "bi male", etc.)
# 
# The Timestamp column immediately catches the eye, i don't need it in this analysis, so i'll drop it.

# In[ ]:


data = data.drop('Timestamp', axis = 1)
data.head()


# In[ ]:


# Shape of dataset
data.shape


# We see, that this dataset is very small, it's a pity, we will have to work with what we have.

# In[ ]:


data.info()


# Also there are NaN values in the dataset, if we will drop them, only 88 entries will remain, so I will leave it as is yet.

# In[ ]:


# Lets take a quicklook at unique values
for col in data.columns.values[1:]:
    print(col, ': ' ,data[col].unique())


# We can notice 4 interesting columns: Server, Role, Class, Race - we need to do something with them because the values contain combinations of different parameters and it is unlikely that it will be possible to extract something useful if we leave it as is.

# In[ ]:


# Let's start with a simple
data['Country'] = data['Country'].str.capitalize()
data['Country'] = data['Country'].replace('Uk', 'U.k')
data[['Gender', 'Sexuality', 'Age', 'Country']].iplot(kind = 'hist', 
                                                      yTitle = 'Count', 
                                                      title = 'Variable distribution',
                                                      subplots = True, 
                                                      shape = (2, 2))


# Wow, I would never have thought that there are so many girls in the WOW, let's try some statistics.

# In[ ]:


def stat(col, z=2.58):
    n = len(data[col])
    
    for i in range(len(data[col].value_counts().index.values)):
        p = data[col].value_counts()[i]/n
        pred = z*np.sqrt((p*(1-p))/n)
        print(data[col].value_counts().index[i],': {0:.2f}% +-{1:.2f}'.format(p*100, pred*100))

stat('Gender')


# In[ ]:


# Same thing for Sexuality and Age
stat('Sexuality')


# In[ ]:


stat('Age')


# Here it is necessary to make a reservation - just two participants aged 43 to 55 toke a part in the poll,so it is not enough to make any conclusions about this category.
# 
# It is also impossible to make any reliable conclusions about the country of the respondents, most of them from the United States, but participants in other countries are represented in the amount of 1, 2, 3 or 4 people, this is clearly not enough for analysis.

# In[ ]:


# Let's dig a little deeper
fig, ax = plt.subplots(1, 3, figsize = (18, 5))
sns.countplot('Sexuality', data = data, hue = 'Gender', ax = ax[0])
sns.countplot('Age', data = data, hue = 'Gender', ax = ax[1])
sns.countplot('Age', data = data, hue = 'Sexuality', ax = ax[2])


# Thus, according to this dataset, we can be 99% sure that 58% +-12.73% of the population of WoW are female players,
# 28% +-11.58% - male and 14% +-8.95% - of the other sex.
# 
# Most players are bisexual - 51.00% +-12.90%, and for the most part these are young people between the ages of 18 and 30.
# Also the largest percentage of non-traditional and bisexual orientation holders is among young people, however, for players aged 31 years and older, the situation is slightly different, among them, hetero orientation prevails.

# In[ ]:


data[['Main', 'Faction', 'Max']].iplot(kind = 'hist', 
                                       yTitle = 'Count', 
                                       title = 'Main\Faction\Max distribution',
                                       subplots = True, 
                                       shape = (1, 3))


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize = (18, 4))
sns.countplot('Main', data = data, hue = 'Gender', ax = ax[0])
sns.countplot('Faction', data = data, hue = 'Gender', ax = ax[1])
sns.countplot('Max', data = data, hue = 'Gender', ax = ax[2])


# From the graphs, we can see the following:
#   - There are more female characters, this is not surprising, since more women than men participated in the survey
#   - Aemale players prefer to create female characters, male players - male characters
#   - Seems that there are more Alliance players than Horde. At the same time the distribution by sex is approximately the same.
#   - Most players have 1-2 110 level characters.
#   - 3 players with 9, 11 and 12 110 lvl characters are male.

# In[ ]:


# Let's look at our most interesting columns
# First - drop Na values
na = data.loc[data['Server'].isna()].index
data = data.drop(na, axis = 0)
data.info()


# In[ ]:


# I want to create dummy variables for every value
def dummies(cols, target):
    for col in cols:
        data[col] = 0
        data.loc[data[target].str.contains(col, regex = False), col] = 1
    data.drop(target, axis = 1, inplace = True)
        
cols = ['PvE', 'PvP', 'RP']
dummies(cols, 'Server')

role = ['DPS', 'Healer', 'Tank']
dummies(role, 'Role')

clas = ['Hunter', 'Druid', 'Priest', 'Shaman', 'Death Knight', 'Demon Hunter', 'Paladin', 'Warlock', 'Warrior',
       'Monk', 'Rogue', 'Mage']
dummies(clas, 'Class')

race = ['Draenei', 'Troll', 'Night Elf', 'Dwarf', 'Blood Elf', 'Tauren', 'Pandaren', 'Gnome', 'Human',
       'Orc', 'Goblin', 'Undead', 'Worgen']
dummies(race, 'Race')


# In[ ]:


# Let's see what we got
data.head()


# Very good, now we can analyze this columns, so let's extract some information from it.

# In[ ]:


s = data[cols + role + clas + race].sum()
fig, ax = plt.subplots(2, 2, figsize = (18, 8))
s[cols].plot(kind = 'bar', ax = ax[0, 0])
s[role].plot(kind = 'bar', ax = ax[0, 1])
s[clas].plot(kind = 'bar', ax = ax[1, 0])
s[race].plot(kind = 'bar', ax = ax[1, 1])


# Now we can see next:
#  - Most players play on PvE servers
#  - Players prefer DPS role
#  - Most popular class - Hunter, least popular - Shaman
#  - Most popular race - Blood Elf, least popular - Gnome

# Next - let's check distribution of this columns by gender

# In[ ]:


serv = data.pivot_table(cols + role + clas + race, ['Gender'], aggfunc = 'sum')

x = ['PvE', 'PvP', 'RP']
y1 = list(serv.loc['Female', ['PvE', 'PvP', 'RP']].values)
y2 = list(serv.loc['Male', ['PvE', 'PvP', 'RP']].values)
y3 = list(serv.loc['Other', ['PvE', 'PvP', 'RP']].values)

trace1 = go.Bar(x = x, 
               y = y1,
               name = 'Female')

trace2 = go.Bar(x = x, 
               y = y2,
               name = 'Male')

trace3 = go.Bar(x = x, 
               y = y3,
               name = 'Other')

dt = [trace1, trace2, trace3]
layout = go.Layout(title = 'Server distribution by gender')
fig = go.Figure(data = dt, layout = layout)
fig.iplot()


# In[ ]:


x = role
y1 = list(serv.loc['Female', role].values)
y2 = list(serv.loc['Male', role].values)
y3 = list(serv.loc['Other', role].values)

trace1 = go.Bar(x = x, 
               y = y1,
               name = 'Female')

trace2 = go.Bar(x = x, 
               y = y2,
               name = 'Male')

trace3 = go.Bar(x = x, 
               y = y3,
               name = 'Other')

dt = [trace1, trace2, trace3]
layout = go.Layout(title = 'Role distribution by gender')
fig = go.Figure(data = dt, layout = layout)
fig.iplot()


# In[ ]:


x = clas
y1 = list(serv.loc['Female', clas].values)
y2 = list(serv.loc['Male', clas].values)
y3 = list(serv.loc['Other', clas].values)

trace1 = go.Bar(x = x, 
               y = y1,
               name = 'Female')

trace2 = go.Bar(x = x, 
               y = y2,
               name = 'Male')

trace3 = go.Bar(x = x, 
               y = y3,
               name = 'Other')

dt = [trace1, trace2, trace3]
layout = go.Layout(title = 'Class distribution by gender')
fig = go.Figure(data = dt, layout = layout)
fig.iplot()


# In[ ]:


x = race
y1 = list(serv.loc['Female', race].values)
y2 = list(serv.loc['Male', race].values)
y3 = list(serv.loc['Other', race].values)

trace1 = go.Bar(x = x, 
               y = y1,
               name = 'Female')

trace2 = go.Bar(x = x, 
               y = y2,
               name = 'Male')

trace3 = go.Bar(x = x, 
               y = y3,
               name = 'Other')

dt = [trace1, trace2, trace3]
layout = go.Layout(title = 'Race distribution by gender')
fig = go.Figure(data = dt, layout = layout)
fig.iplot()


# Well, I think that I'll stop on this. I'm realy like this dataset, but it have one very big disadvandage - it's extremelly small, and it quite imbalanced, so we cant trust the conclusions that I made here. There is only one solution for this problem - we need more data.
