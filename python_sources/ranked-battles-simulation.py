#!/usr/bin/env python
# coding: utf-8

# Anyone who has played online games has probably come across 'ranked battles'. This is a mode where you play individually or in smaller teams and where you aim to move up from a starter rank to the top rank.
# 
# This is a simulation based on the rules used for WarGaming's Season 1 Ranked Battles for *World of Warships Legends*, which is on console. The rules are as follows:
# 
# * The player starts at Rank 10 and is aiming to reach Rank 1 
# * Each rank has a set number of stars that you need in order to move up to the next rank
# * Win a game and you receive a star. Win enough stars and you will be promoted to the next rank
# * Lose a game and you lose a star. However, you don't lose a star if you finished top of your team in terms of xp earned
# * Some ranks are irrevocable i.e. once reached you cannot be demoted from them
# 
# As you can tell, Ranked Battles have a *Snakes and Ladders* feel about them, which often causes player stress and rage!
# 
# Based on the rules above, this simulation uses basic random number generation and 1000 simulated runs to produce a distribution curve for how many games you will have to play in order to reach Rank 1. The better players have typically reported requiring only 60 games, which according to this simulation requires a Win-Rate of 60% (and finishing top of your team 50% of the time when losing). An average player with a 50% Win-Rate is going to have to play on average 130+ games! However, the distribution has a long tail which means some players just won't be able to play enough games to get there....
# 
# Feel free to play and adjust. Pass pack any suggestions to improve my hacky code :)

# In[ ]:


import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Adjust your win-rate and likelihood of finishing top of the team. 
# A number of '50' will be treated as '50%' so no need to add '%'.

# In[ ]:


win_rate = 60
top_rate = 50


# Here you can adjust how many stars are needed for each rank and which ranks are irrevocable.

# In[ ]:


stars_needed = {10:2,9:2,8:2,7:3,6:3,5:3,4:4,3:4,2:4}
irrev_rank = {10:True, 9:True, 8:False,7:True,6:False,5:False, 4:True, 3:False, 2:False, 1:True}


# The rest of the code is here:

# In[ ]:


title = "Win Rate: " + str(win_rate) + "%, Finish Top: " + str(top_rate) + "%"
results = []

#1,000 simulations
for x in range(1000):
    rank = 10
    stars = 0
    games = 0
    while rank > 1:
        games = games + 1
        n = random.randrange(1,100)
        #Loss or draw
        if n > win_rate:
            #second chance if top on xp
            n = random.randrange(1,100)
            if n > top_rate:
                #if not you lose a star
                stars = stars - 1
                if stars <0:
                    #irrevocable rank so stay on that rank
                    if irrev_rank[rank]:
                        stars = 0
                    #otherwise demoted
                    else:
                        rank = rank + 1
                        stars = stars_needed[rank]-1
        #Win
        else:
            stars = stars + 1
            if stars >= stars_needed[rank]:
                stars = 0
                rank = rank - 1

    results.append(games)

#calculate the average
mn = round(np.mean(results))
average = 'Average (' + str(int(mn)) + ')'

#Plotting
plt.figure(figsize=(12,6))
plt.grid()
plt.xlabel('Total Games Played')
plt.title(title)
sns.kdeplot(results,shade=True)
plt.annotate(average,xy = (mn,0),xytext=(mn+10,0.005),
    arrowprops=dict(facecolor='black',shrink=0.05))
plt.tight_layout()
plt.show()

