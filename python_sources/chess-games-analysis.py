#!/usr/bin/env python
# coding: utf-8

# # Chess games analysis

# ## Table of contents
# * [About the data](#About-the-data)
# * [Loading and cleaning the data](#Loading-and-cleaning-the-data)
# * [Basic observations](#Basic-observations)
# * [Types of victory](#Types-of-victory)
# * [Number of turns to mate](#Number-of-turns-to-mate)
# * [Scholar mates](#Scholar-mates)
# * [Wins by lower rated players](#Wins-by-lower-rated-players)
# * [Squares occupation](#Squares-occupation)

# # About the data
# The dataset contains most recent (at the time of creating) ~20k games rated/non-rated from [lichess.org](http://lichess.org).
# 
# I'm analysing only the rated games, of which there are 16,155.

# # Loading and cleaning the data

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re

sns.set(color_codes=True, style='darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


games = pd.read_csv('../input/games.csv')
games.head(2)


# In[ ]:


games = games[games.rated]  # only rated games
games['mean_rating'] = (games.white_rating + games.black_rating) / 2
games['rating_diff'] = abs(games.white_rating - games.black_rating)


# # Basic observations

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(games['mean_rating'])


# **Most games were played between ~1500 rated players**
# 

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(games.turns)


# **and took around 50 turns.**

# # Types of victory

# In[ ]:


games.victory_status.value_counts()


# In[ ]:


under_1500 = games[games.mean_rating < 1500]
under_2000 = games[games.mean_rating < 2000]
over_2000 = games[games.mean_rating > 2000]

brackets = [under_1500, under_2000, over_2000]
bracket_titles = ['Under 1500', 'Under 2000', 'Over 2000']


# In[ ]:


plt.figure(figsize=(15,11))
for i, bracket in enumerate(brackets):
    victory_status = bracket.victory_status.value_counts()
    plt.subplot(1, 4, i+1)
    plt.title(bracket_titles[i])
    plt.pie(victory_status, labels=victory_status.index)


# **Most of the games ended by resignation (more than 50%). 
# As we can see, players resign often in higher rated games. I suppose these players are certain if the have disadvantage their opponent is not going to have trouble mating them - they don't waste their time playing lost games.**
# 
# **In games over 2000 there are also a little more draws - players make less mistakes.**

# # Number of turns to mate

# In[ ]:


mate_games = games[games.victory_status=='mate']

under_1500 = mate_games[mate_games.mean_rating < 1500]
under_2000 = mate_games[mate_games.mean_rating < 2000]
over_2000 = mate_games[mate_games.mean_rating > 2000]

m_brackets = [under_1500, under_2000, over_2000]


# In[ ]:


turn_means = [b.turns.mean() for b in m_brackets]

plt.figure(figsize=(10,5))
plt.ylim(0, 100)
plt.title('Number of turns until mate')
plt.plot(bracket_titles, turn_means, 'o-', color='r')


# In[ ]:


plt.figure(figsize=(10,5))
plt.scatter(mate_games.mean_rating, mate_games.turns)


# **Even though there doesn't appear to be strong correlation between players' rating and number of turns to mate, in higher rated games the number is slightly higher.**

# In[ ]:


mate_games.loc[mate_games['turns'].idxmax()]


# **The longest game resulting in mate (222 moves) were played by players with ratings 1617 and 1614.**

# # Scholar mates

# In[ ]:


scholar_mates = mate_games[mate_games.turns==4]
scholar_mates


# **As I tried to see if there were any [Scholar's Mates](https://en.wikipedia.org/wiki/Scholar%27s_mate) (mate in 4 moves). However, I only found one player boosting his rating beating his other accounts.**
# 
# **Checking his  [profile on lichesss](https://lichess.org/@/SMARTduckduckcow) we can see he's banned now.**

# # Wins by lower rated players

# In[ ]:


white_upsets = games[(games.winner == 'white') & (games.white_rating < games.black_rating)]
black_upsets = games[(games.winner == 'black') & (games.black_rating < games.white_rating)]
upsets = pd.concat([white_upsets, black_upsets])


# In[ ]:


THRESHOLD = 900
STEP = 50

u_percentages = []

print(f'Rating difference : Percentage of wins by weaker player')
for i in range(0+STEP, THRESHOLD, STEP):
    th_upsets = upsets[upsets.rating_diff > i]
    th_games = games[games.rating_diff > i]
    upsets_percentage = (th_upsets.shape[0] / th_games.shape[0]) * 100
    u_percentages.append([i, upsets_percentage])
    print(f'{str(i).ljust(18)}:  {upsets_percentage:.2f}%')


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(*zip(*u_percentages))
plt.xlabel('rating difference')
plt.ylabel('upsets percentage')


# **Chances of the weaker player winning against higher rated player are, naturally, decreasing as the rating difference increases.
# However, approximately every fourth game where the ranking difference is more than 100 points results in a win of lower-ranked player - I expected this number to be a little less.**

# # Squares occupation

# In[ ]:


p = re.compile('([a-h][1-8])')
squares = {}
for moves in games.moves:
    for move in moves.split():
        try:
            square = re.search(p, move).group()
        except AttributeError:  # castling
            square = move.replace('+', '')
        squares[square] = squares.get(square, 0) + 1


# In[ ]:


squares_df = pd.DataFrame.from_dict(squares, orient='index', columns=['count'])

# add castling

total_shorts = int(squares_df.loc['O-O'])
total_longs = int(squares_df.loc['O-O-O'])

half_shorts = total_shorts//2
half_longs = total_longs//2

# white short castling
squares_df.loc['f1'] = squares_df.loc['f1'] + half_shorts
squares_df.loc['g1'] = squares_df.loc['g1'] + half_shorts
# black short castling
squares_df.loc['f8'] = squares_df.loc['f8'] + half_shorts
squares_df.loc['g8'] = squares_df.loc['g8'] + half_shorts 
# white long castling
squares_df.loc['c1'] = squares_df.loc['c1'] + half_longs
squares_df.loc['d1'] = squares_df.loc['d1'] + half_longs
# black long castling
squares_df.loc['c8'] = squares_df.loc['c8'] + half_longs
squares_df.loc['d8'] = squares_df.loc['d8'] + half_longs

squares_df.drop(['O-O', 'O-O-O'], inplace=True)


# ### Interlude: Which castling is more common?

# In[ ]:


total_castles = total_shorts + total_longs
print(f'Short: {(total_shorts/total_castles)*100:.2f}%')
print(f'Long: {(total_longs/total_castles)*100:.2f}%')


# In[ ]:


plt.figure(figsize=(10,5))
plt.pie([total_shorts, total_longs],
       labels=['Short', 'Long'])


# In[ ]:


squares_df.reset_index(inplace=True)
squares_df['letter'] = squares_df['index'].str[0]
squares_df['number'] = squares_df['index'].str[1]


# In[ ]:


squares_df = squares_df.pivot('number', 'letter', 'count')
squares_df.sort_index(level=0, ascending=False, inplace=True)  # to get right chessboard orientation
squares_df


# In[ ]:


sns.set(rc={'figure.figsize':(20,15)})
hm = sns.heatmap(squares_df,
            cmap='Oranges',
            annot=False, 
            vmin=0,
            fmt='d',
            linewidths=2,
            linecolor='black',
            cbar_kws={'label':'occupation'},
            )
hm.set(xlabel='', ylabel='')

