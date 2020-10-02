#!/usr/bin/env python
# coding: utf-8

# # Fifa 2k19 Data Analysis
# Getting started with data anslysis is a lot easier if the data that you are analysing is intresting to you. You can't stop analysing data of a game that you play.
# 
# With some question that I had and some questions that I came up with during the analysis, I've tried to come up with resonable answers to those questions.
# 
# # Contents
# * Load data
# * View a small sample of data
# * Correlation heatmap
# * Height vs dribbling Skills
# * Weight vs dribbling Skills
# * **L. Messi vs Ronaldo**
# * Top 10 Players based on Overall skills
# * Top 5 clubs with overall best player
# * Age distribution of players in the clubs
# * Age distribution of players in countries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Visualization
import seaborn as sns # Visualization
from IPython.display import display, HTML # IPython notebook display
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
# Any results you write to the current directory are saved as output.


# # Load data
# * Load data from the input file.
# 
# * Get information about the data.

# In[ ]:


data = pd.read_csv("../input/data.csv")
data.info()


# # View a small sample of the data
# It's always good to visualize the data to get a good idea of what we are dealing with. 

# In[ ]:


data.head()


# # Correlation heatmap
# Let's first draw correlation heatmap and try to figure out some correlations.
# That way we neglect a lot of unnecessary relations and find out some good ones.

# In[ ]:


plt.rcParams['figure.figsize']=(25,16)
hm=sns.heatmap(data[['Age', 'Overall', 'Potential', 'Special',
    'Body Type', 'Position',
    'Height', 'Weight', 'Crossing',
    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
    'Marking', 'StandingTackle', 'SlidingTackle']].corr(), annot = True, linewidths=.5, cmap='Blues')
hm.set_title(label='Heatmap of dataset', fontsize=20)


# # Height vs dribbling Skills
# * We found that players who are **shorter are better at dribbling.**
# 
# **Note: The two dips on the 2nd and 3rd bars are because the height is given in inches and sorting gets 5'10 and 5'11 before 5'2 .**

# In[ ]:


plt.xlabel('Height', fontsize=20)
plt.ylabel('Dribbling', fontsize=20)
plt.title('Height vs Dribbling', fontsize = 25)
sns.barplot(x='Height', y='Dribbling', data=data.sort_values('Height', inplace=False), alpha=0.6)


# # Weight vs dribbling skills
# * With some exceptions we can say that **lighter** players are better at dribbling.

# In[ ]:


plt.xlabel('Weight', fontsize=20)
plt.ylabel('Dribbling', fontsize=20)
plt.title('Weight vs Dribbling', fontsize = 25)
sns.barplot(x='Weight', y='Dribbling', data=data.sort_values('Weight'),alpha=0.6)


# # L. Messi Vs Cristiano Ronaldo
# ## Time to figure out who's better at what.
# <div>
# <img style="float: left;" src="https://cdn.sofifa.org/players/4/19/158023.png">
#     <img style="float: left;" src="https://cdn.sofifa.org/teams/2/light/241.png">	
# <h3>L.  Messi</h3>
# </div>
# <br>
# <ul>
#   <li>   Short Passing
#     <li> Dribbling
#     <li> Curve
#     <li> FK Accuracy
#     <li> Long Passing
#     <li> Vision
#         </ul>
# <div>
# <img style="float: left;" src="https://cdn.sofifa.org/players/4/19/20801.png">
#     <img style="float: left;" src="https://cdn.sofifa.org/teams/2/light/45.png">
# <h3>C. Ronaldo</h3>
# <div>
#     <br>
#     <ul>
#     <li> Heading Accuracy
#     <li> Shot Power
#     <li> Jumping
#     <li> Stamina
#     <li> Strength
#     <li> Penalties
#     </ul>

# In[ ]:


skills = ['Overall', 'Potential', 'Crossing',
   'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
   'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
   'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
   'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
   'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
   'Marking', 'StandingTackle', 'SlidingTackle']


# In[ ]:


messi = data.loc[data['Name'] == 'L. Messi']
messi = pd.DataFrame(messi, columns = skills)
ronaldo = data.loc[data['Name'] == 'Cristiano Ronaldo']
ronaldo = pd.DataFrame(ronaldo, columns = skills)

plt.figure(figsize=(14,8))
sns.pointplot(data=messi,color='blue',alpha=0.6)
sns.pointplot(data=ronaldo, color='red', alpha=0.6)
plt.text(5,55,'Messi',color='blue',fontsize = 25)
plt.text(5,50,'Ronaldo',color='red',fontsize = 25)
plt.xticks(rotation=90)
plt.xlabel('Skills', fontsize=20)
plt.ylabel('Skill value', fontsize=20)
plt.title('Messi vs Ronaldo', fontsize = 25)
plt.grid()


# # Top 10 Players based on Overall skills

# In[ ]:


display(
    HTML(data.sort_values('Overall', ascending=False)[['Name', 'Overall']][:10].to_html(index=False)
))


# # Top 5 clubs with overall best player

# In[ ]:


top_clubs = data.groupby(['Club'])['Overall'].max().sort_values(ascending = False)
top_clubs.head(5)


# # Age distribution of players in the clubs
# **Lets see which teams are getting older and will need fresh legs soon.**
# * Well we know **Juventus** and **FC Barcelona** need to find some fresh legs. Ronaldo and Messi won't be there forever.** 
# 

# In[ ]:


top_club_names = ('FC Barcelona', 'Juventus', 'Paris Saint-Germain', 'Chelsea', 'Manchester City')
clubs = data.loc[data['Club'].isin(top_club_names) & data['Age']]
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
ax = sns.boxenplot(x="Club", y="Age", data=clubs)
ax.set_title(label='Age distribution in the top 5 clubs', fontsize=25)
plt.xlabel('Clubs', fontsize=20)
plt.ylabel('Age', fontsize=20)
plt.grid()


# # Age distribution of players in countries
# 
# * Argentina seems to have a wider span in age distribution. Time for the old guns to make way for the new ones.
# * Neatherla[](http://)nds, Germany, Belgium, and France seems to have a team on younger side.

# In[ ]:


countries_names = ('France', 'Brazil', 'Germany', 'Belgium', 'Spain', 'Netherlands', 'Argentina', 'Portugal', 'Chile', 'Colombia')
countries = data.loc[data['Nationality'].isin(countries_names) & data['Age']]
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
ax = sns.boxenplot(x="Nationality", y="Age", data=countries)
ax.set_title(label='Age distribution in countries', fontsize=25)
plt.xlabel('Countries', fontsize=20)
plt.ylabel('Age', fontsize=20)
plt.grid()

