#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# My first Python notebook! Board gaming has developed into a huge hobby of mine, and I'm really interested in exploring some statistics from the board game community! We are going to look into the BoardGameGeek data set, which has been collected together and cleaned by members of the Kaggle community.
# 
# Below, I take a look at how the rankings for board games are influenced by the number of voters. This notebook also takes a dive into some of the categorical data. We even show that board games with minatures are statistically given higher ratings that board games without miniatures! 
# 
# First things first, we have to get all the data imported in, and ready to rock. And let's see what the data we can even look at.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
bgg_data = pd.read_csv('../input/bgg_db_2017_04.csv', header=0, encoding='latin1', index_col=0)
bgg_data.head()


# What kind of data comes with each board game?  

# In[ ]:


bgg_data.columns


# It looks like we've got everything from the board game name, and features (player counts, times, release dates), to user supplied data like the ratings, number of votes, number of owners.  We also have some interesting data in the categories and mechanics, that we will look into later.  
# 
# # Looking at Average Ratings and Geek Ratings
# 
# First, let's see how well the average user rating tracks with the Geek Ratings, which tie directly to the ratings!

# In[ ]:


fig1, ax1 = plt.subplots(figsize=(10,6))
bgg_data.plot(x='avg_rating', y='geek_rating', ax=ax1, kind='scatter')
myX = [5.0,10.0]
myY = [5.0,10.0]
ax1.plot(myX, myY, color='red')
plt.xlim([5.55,9.5])
plt.ylim([5.55,8.5])
plt.show()


# Hmm, there's a huge spread there at lower number of geek ratings. Also interestingly, the geek rating never goes above the average rating (supported by information from BGG's FAQ). Let's see if we can find any more information out, by teasing out some information using the ownership and votereship data.  
# 
# Essentially, we are going to look at each category of board games performs based on how many people own the game.  We are going to categorize games into 7 bins (less than or equal to):
# 
# * 10%
# * 25%
# * 50%
# * 75%
# * 90%
# * 99% 
# 
# most owned / voted on games.  

# In[ ]:


# Set the quantiles & the respective colors
quant_cat = ['0-10%','10-25%','25-50%','50-75%','75-90%','90 - 99%', '>99%']
colors = {'0-10%':'red', '10-25%':'blue', '25-50%':'green', '50-75%':'brown', 
         '75-90%':'orange','90 - 99%':'grey', '>99%':'black'}

# split by ownership!
bgOwnedQuant = bgg_data['owned'].quantile([0,0.1, 0.25, 0.5, 0.75, 0.9,0.99,1]).tolist()
bgg_data['ownership'] = pd.cut(bgg_data['owned'], bgOwnedQuant, labels=quant_cat, 
                               include_lowest=True, right=True).astype('category')
# split by votership
bgVotesQuant = bgg_data['num_votes'].quantile([0,0.1, 0.25, 0.5, 0.75, 0.9,0.99,1]).tolist()
bgg_data['votership_turnout'] = pd.cut(bgg_data['num_votes'], bgVotesQuant, labels=quant_cat, 
                               include_lowest=True, right=True).astype('category')

# Make those plots
fig, (ax1, ax2) = plt.subplots(2, figsize=(12,16))
ax1.set_title('Geek Rating vs Average Rating by Ownership Quantiles')
for label, group in bgg_data.groupby('ownership'):
    group.plot(x='avg_rating', y='geek_rating', kind="scatter", 
               marker='o', alpha=0.5, ax=ax1, color=colors[label], label=label)
ax1.plot(myX, myY, color='red')  
ax1.set_xlabel('Avg Rating')
ax1.set_ylabel('Geek Rating')

ax2.set_title('Geek Rating vs Average Rating by Votership Quantiles')
for label, group in bgg_data.groupby('votership_turnout'):
    group.plot(x='avg_rating', y='geek_rating', kind="scatter", 
               marker='o', alpha=0.5, ax=ax2, color=colors[label], label=label)
ax2.set_xlabel('Avg Rating')
ax2.plot(myX, myY, color='red')
ax2.set_ylabel('Geek Rating')
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
ax1.set_xlim([5.55,9.5])
ax2.set_xlim([5.55,9.5])
ax1.set_ylim([5.55,8.5])
ax2.set_ylim([5.55,8.5])
plt.show()


# Wow!  That's super cool.  You can definitely see that the more people that own the game, the closer the average rating approaches the Geek Rating.  But what's even cleaner is separation by the number of voters! This makes sense, since the Geek Rating uses some secret algorithm to add a certain number of ratings to each game (of approximately average rating).  
# 
# # Voters / Owners, Game Mechanics and Categories
# 
# Now, let's see if we can find anything cool within the different categories of board games!  We are going to define a function to extract the categories for board games.  This will be super useful later!

# In[ ]:


bgg_data['penetration_ratio'] = bgg_data['num_votes'] / bgg_data['owned']
bgg_most_rated_data = bgg_data.copy()
#bgg_most_rated_data = bgg_data[bgg_data['num_votes'] >= 1000]
#bgg_most_rated_data = bgg_most_rated_data[bgg_most_rated_data['votership_turnout'] == ('>99%' or '90-99%' or '75-90%' or '50-75%')]
fig3, (ax3_1, ax3_2) = plt.subplots(2, figsize=(16,12))
ax3_1.set_title('Penetration Ratio as a function of Ownership')
ax3_2.set_title('Penetration Ratio as a function of Votership')
for label, group in bgg_most_rated_data.groupby('ownership'):
    group.plot(x='owned', y='penetration_ratio', kind="scatter", 
               marker='x', alpha=0.5, ax=ax3_1, color=colors[label], label=label)
for label, group in bgg_most_rated_data.groupby('votership_turnout'):
    group.plot(x='num_votes', y='penetration_ratio', kind="scatter", 
               marker='x', alpha=0.5, ax=ax3_2, color=colors[label], label=label)
more_voters_Y = [1.0, 1.0]
more_voters_X = [5.5, 110000]
ax3_1.plot(more_voters_X, more_voters_Y, color='red')
ax3_2.plot(more_voters_X, more_voters_Y, color='red')
ax3_1.set_xlabel('Number of Owners')
ax3_1.set_ylabel('Penetration Ratio')
ax3_1.set_xscale('log')
ax3_1.set_xlim([40, 110000])
ax3_1.set_ylim([0,2])
ax3_2.set_xlabel('Number of Ratings')
ax3_2.set_ylabel('Penetration Ratio')
ax3_2.set_xscale('log')
ax3_2.set_xlim([40, 110000])
ax3_2.set_ylim([0,2])
plt.show()


# Well, this is something particularly interesting.  With very few exceptions, most games have a larger number of owners than ratings!   This could be because people use BGG as a medium to track their collection, or that people have gotten the games but haven't played them yet... (Something chronically lamented in the hobby!)
# 
# Let's take a look below at the games with the highest penetration ratios; they are mostly card games like Hearts, Spaces, Bridge, Euchre, or poker. It's kind of interesting, can you saw you have hearts, or spades in your collection?  Certainly you do, if you have a deck of standard cards.  Another one in the top is crokinole, which is notoriously large and not as easy to acquire! 
# 

# In[ ]:


bgg_data[bgg_data['penetration_ratio'] >= 1.0].loc[bgg_data['num_votes'] >= 1000, 
                                                   ['names', 'year', 'geek_rating', 'num_votes', 'penetration_ratio']].sort_values(by='penetration_ratio', ascending=False).head()


# Now, let's see if we can find anything cool within the different categories of board games!  We are going to define a function to extract the categories for board games.  This will be super useful later!

# In[ ]:


def get_categorical_data(series):#Fuction for extracting set of categorical data labels
    category_names = series.apply(lambda s:s.split(','))
    category_names = category_names.tolist()
    all_the_categories = []
    for game in category_names:
        for item in game:
            all_the_categories.append(item.replace('\n', ' ').replace('/', '-').strip())
    return set(all_the_categories)


# In[ ]:


category_set = get_categorical_data(bgg_data['category'])
mechanics_set = get_categorical_data(bgg_data['mechanic'])

{'Category':category_set, 'Mechanics':mechanics_set}


# In[ ]:


bgg_mechanics_data = bgg_data.loc[:, ['names', 'year', 'mechanic', 'geek_rating', 'avg_rating']] 
#for mech in sorted(list(mechanics_set)):
#    bgg_mechanics_data[mech] = np.zeros(len(bgg_mechanics_data.index))
for game in bgg_mechanics_data.index.tolist():
    game_mechs = bgg_mechanics_data.loc[game, 'mechanic'].split(',')
    game_mechs = [s.replace('\n', ' ').replace('/', '-').strip() for s in game_mechs]
    for mech in game_mechs:
        bgg_mechanics_data.loc[game, mech] = bgg_mechanics_data.loc[game, 'avg_rating']

#bgg_mechanics_data.describe().T.sort_values(by='count', ascending=False)
fig4, ax4 = plt.subplots(figsize=(22,10))
bgg_mechanics_data.boxplot(sorted(list(mechanics_set)), ax=ax4, showmeans=True)
ax4.set_xticklabels(sorted(list(mechanics_set)), rotation=40, ha='right')
plt.show()


# There's a ton of interesting information in these plots.  Let's try to slim down the information by looking at categories with at least 200 games rated.  

# In[ ]:


bgg_top_mechanics_data = bgg_mechanics_data[bgg_mechanics_data.columns[bgg_mechanics_data.count()>200]]

top_mechanics = bgg_top_mechanics_data.columns.values.tolist()[4:]
fig5, ax5 = plt.subplots(figsize=(22,10))
bgg_top_mechanics_data.boxplot(top_mechanics, ax=ax5, showmeans=True)
ax5.set_xticklabels(top_mechanics, rotation=40, ha='right')
plt.show()


# So, if we check these top mechanics, we will note there many of them have significant positive outliers but only the deckbuilding category has significant negative outliers.  The simulation & hex-and-counter games have a have a fairly high median rating, as does deck-pool building.  Interestingly, dice rolling has one of the largest spans of ratings, along with variable player powers, cooperative, and area movement games.  

# In[ ]:


bgg_top_mechanics_data.describe().T.sort_values(by='mean', ascending=False)


# In[ ]:


bgg_cat_data = bgg_data.loc[:, ['names', 'year','owned', 'category','geek_rating', 'avg_rating' ]] 
for game in bgg_cat_data.index.tolist():
    game_cats = bgg_cat_data.loc[game, 'category'].split(',')
    game_cats = [s.replace('\n', ' ').replace('/', '-').strip() for s in game_cats]
    for cat in game_cats:
        bgg_cat_data.loc[game, cat] = bgg_cat_data.loc[game, 'avg_rating']

bgg_top_cat_data = bgg_cat_data[bgg_cat_data.columns[bgg_cat_data.count()>200]]

top_cat = bgg_top_cat_data.columns.values.tolist()[5:]
fig6, ax6 = plt.subplots(figsize=(22,10))
bgg_top_cat_data.boxplot(top_cat, ax=ax6, showmeans=True)
ax6.set_xticklabels(top_cat, rotation=40, ha='right')
plt.show()


# In[ ]:


bgg_top_cat_data.describe().T


# Where the type of mechanics in the game did not see to have an effect on the game, the categories the game falls under does have a significant impact!  You can clearly see that the average game with miniatures is rated higher than most other categories. 
# 
# Also of interest is the slight boost in ratings for games themed around World War II.  We see that wargames also have a slightly higher average rating, whereas humor games have a much lower average rating.  
# 
# # Minis, or not??
# 
# Let's look and see how games with and without miniatures are rated.

# In[ ]:


import math 
from scipy.stats import ks_2samp
miniatures_comp = bgg_cat_data.loc[:, ['names', 'year','owned', 'category','geek_rating', 'avg_rating', 'Miniatures' ]] 
for game in miniatures_comp.index.tolist():
    #print(game['Miniatures'])
    if math.isnan(miniatures_comp.loc[game, 'Miniatures']) :
        miniatures_comp.loc[game, 'No Miniatures'] = miniatures_comp.loc[game,'avg_rating']
# miniatures_comp[['Miniatures', 'No Miniatures']].describe()

fig7, ax7 = plt.subplots()
mini_heights, mini_bins = np.histogram(miniatures_comp['Miniatures'].dropna(axis=0), bins=50)
nomini_heights, nomini_bins = np.histogram(miniatures_comp['No Miniatures'].dropna(axis=0), bins=mini_bins)
mini_heights = mini_heights / miniatures_comp['Miniatures'].count()
nomini_heights = nomini_heights / miniatures_comp['No Miniatures'].count()
width = (mini_bins[1] - mini_bins[0])/3
ax7.bar(mini_bins[:-1], mini_heights, width=width, facecolor='cornflowerblue', label='Minis')
ax7.bar(nomini_bins[:-1]+width, nomini_heights, width=width, facecolor='seagreen', label='No Minis')
ax7.set_xlabel('Average Rating')
ax7.set_ylabel('Rating Denisty')
ax7.legend(loc='upper right')
plt.show()

# do Kolmogorov - Smirnov analysis of null hypothesis

print(ks_2samp(miniatures_comp['Miniatures'].dropna(axis=0),
               miniatures_comp['No Miniatures'].dropna(axis=0)))


# Using the Kolmogorov-Smirov test, it looks like we cannot reject the null hypothesis. The distributions appear to be statistically different! An extremely small p-value of 9.7e-21%, a KS D statistic of 0.29 shows the max distance between the CDFS for the two distributions is 0.29 (in the average rating). Coupled with the extremely low p value, this result suggests that games with Miniatures are statistically rated higher!     

# More to come soon!!
