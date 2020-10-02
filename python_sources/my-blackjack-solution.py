#!/usr/bin/env python
# coding: utf-8

# **This is my try to the blackack challenge proposed in the learn pyhton course**
# 
# ![image.png](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQD-t878iKbW0m90Shc5N_YzVLZ6lrbV8F_aK6WeRW5QHty_htO)
# 
# **The final result and the win rate are over 1,000,000 games(so we have pretty solid conclusions).
# The fact that the dealer wins a higher percentage of the games shows us that, even playing close to the statistic perfection, the casino will always win.**

# 

# In[ ]:


from learntools.core import binder; binder.bind(globals())
from learntools.python.ex3 import q7 as blackjack

def should_hit(player_total, dealer_total, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    if player_total <=11 and dealer_total >0 and player_aces>0:
        return True
    elif player_total <=0 and dealer_total >0 and player_aces==1:
        return True
    elif player_total <= 14 and (dealer_total < 3 or dealer_total > 7):
        return True
    else:
        return False

blackjack.simulate(n_games=1000000)
    


# **Please vote up the kernel and comment your tries at this challenge =)**
