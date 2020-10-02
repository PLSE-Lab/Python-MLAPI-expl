#!/usr/bin/env python
# coding: utf-8

# Using code from Kaggle exercises to create an optimal strategy to playing Blackjack verses the dealer. 

# In[ ]:


from learntools.core import binder; binder.bind(globals())
from learntools.python.ex3 import q7 as blackjack


# Most conservative strategy (never hit)

# In[ ]:


def should_hit(player_total, dealer_card_val, player_aces):
    return False


# In[ ]:


blackjack.simulate_one_game()


# In[ ]:


blackjack.simulate(n_games=90000)


# Adjust should_hit function to determine when the player should hit based on: dealer initial card value, number of aces (the hard() function) and player total. 

# In[ ]:


def hard(player_aces):
    if player_aces >= 1:
        return True
    else: return False



def should_hit(player_total, dealer_card_val, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    if hard(player_aces)== True and player_total <= 11: return True
    elif hard(player_aces)== True and player_total >= 18: return False
    elif hard(player_aces)== False and player_total <= 15: return True
    elif hard(player_aces)== False and player_total >= 17: return False
    elif hard(player_aces)== True and (4 <= dealer_card_val <= 6) and player_total == 12: return False
    elif hard(player_aces)== True and (4 <= dealer_card_val <= 6) and (13 <= player_total <= 16): return False
    elif hard(player_aces)== False and [(dealer_card_val != 9) or (dealer_card_val != 10) or (dealer_card_val != 11) or (dealer_card_val != 1)]: return False
    else: return True


# Simulate 1 game:

# In[ ]:


blackjack.simulate_one_game()


# Simulate many games:

# In[ ]:


blackjack.simulate(n_games=90000)


# Defining functions for basic card counting 

# In[ ]:


def count(player_total):
    if player_total <= 8: return 2
    elif 8< player_total < 13: return 1
    elif player_total >= 17: return -2
    else: return 0
 

def count_d(dealer_card_val):
    if dealer_card_val >= 9: return -1
    elif dealer_card_val < 7: return 1
    else: return 0
    
    
def tot(player_total,dealer_card_val):
    return count_d(dealer_card_val) + count(player_total)


# First we'll see how the card counting strategy does on it's own; then we'll add it to the previous strategy

# In[ ]:


def should_hit(player_total, dealer_card_val, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    if tot(player_total,dealer_card_val) > 0: return True
    if tot(player_total,dealer_card_val) < 0: return False
    elif tot(player_total,dealer_card_val) == 0: return True


# Simulate 1 game:

# In[ ]:


blackjack.simulate_one_game()


# Simulate many games:

# In[ ]:


blackjack.simulate(n_games=90000)


# Surprisingly, the simplified card counting strategy does basically as well as the 'optimal strategy' defined above. Next, we'll combine both methods. 

# In[ ]:


def should_hit(player_total, dealer_card_val, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    if hard(player_aces)== True and player_total <= 11: return True
    elif hard(player_aces)== True and player_total >= 18: return False
    elif hard(player_aces)== False and player_total <= 15: return True
    elif hard(player_aces)== False and player_total >= 17: return False
    elif hard(player_aces)== True and (4 <= dealer_card_val <= 6) and player_total == 12: return False
    elif hard(player_aces)== True and (4 <= dealer_card_val <= 6) and (13 <= player_total <= 16): return False
    elif hard(player_aces)== False and [(dealer_card_val != 9) or (dealer_card_val != 10) or (dealer_card_val != 11) or (dealer_card_val != 1)]: return False
    elif tot(player_total,dealer_card_val) > 0: return True
    elif tot(player_total,dealer_card_val) < 0: return False
    elif tot(player_total,dealer_card_val) ==0: return True
    else: return False


# Again, preforming simulations:

# In[ ]:


blackjack.simulate_one_game()


# The combined model fares comparably to the 'optimal' strategy (but is better than the simplified counting cards strategy implemented, as one would expect). 

# In[ ]:


blackjack.simulate(n_games=90000)


# What have we learned? Its actually much harder to optimize the most conservative strategy (never hitiing); even by encorporating some version of counting cards or by using the most optimal strategy as suggested by experts. 
