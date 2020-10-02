#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# Ready for a quick test of your logic and programming skills?
# 
# In today's micro-challenge, you will write the logic for a blackjack playing program.  Our dealer will test your program by playing 50,000 hands of blackjack. You'll see how frequently your program won, and you can discuss how your approach stacks up against others in the challenge.
# 
# ![Blackjack](http://www.hightechgambling.com/sites/default/files/styles/large/public/casino/table_games/blackjack.jpg)

# # Blackjack Rules
# 
# We'll use a slightly simplified version of blackjack (aka twenty-one). In this version, there is one player (who you'll control) and a dealer. Play proceeds as follows:
# 
# - The player is dealt two face-up cards. The dealer is dealt one face-up card.
# - The player may ask to be dealt another card ('hit') as many times as they wish. If the sum of their cards exceeds 21, they lose the round immediately.
# - The dealer then deals additional cards to himself until either:
#     - The sum of the dealer's cards exceeds 21, in which case the player wins the round, or
#     - The sum of the dealer's cards is greater than or equal to 17. If the player's total is greater than the dealer's, the player wins. Otherwise, the dealer wins (even in case of a tie).
# 
# When calculating the sum of cards, Jack, Queen, and King count for 10. Aces can count as 1 or 11 (when referring to a player's "total" above, we mean the largest total that can be made without exceeding 21. So e.g. A+8 = 19, A+8+8 = 17)
# 
# # The Blackjack Player
# You'll write a function representing the player's decision-making strategy. Here is a simple (though unintelligent) example.
# 
# **Run this code cell** so you can see simulation results below using the logic of never taking a new card.

# In[ ]:


def should_hit(player_total, dealer_card_val, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    return False


# We'll simulate games between your player agent and our own dealer agent by calling your function. So it must use the name `should_hit`

# # The Blackjack Simulator
# 
# Run the cell below to set up our simulator environment:

# In[ ]:


# SETUP. You don't need to worry for now about what this code does or how it works. 
# If you're curious about the code, it's available under an open source license at https://github.com/Kaggle/learntools/
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex3 import q7 as blackjack
print('Setup complete.')


# Once you have run the set-up code. You can see the action for a single game of blackjack with the following line:

# In[ ]:


blackjack.simulate_one_game()


# You can see how your player does in a sample of 50,000 games with the following command:

# In[ ]:


blackjack.simulate(n_games=50000)


# # Your Turn
# 
# Write your own `should_hit` function in the cell below. Then run the cell and see how your agent did in repeated play.

# ## Extra libs

# In[ ]:


from random import choice
from collections import Counter, defaultdict
from functools import lru_cache
from decimal import Decimal

# I decided to calculate probabilities using Decimal type following
# the comment https://www.kaggle.com/learn-forum/58735#442773
# Yet, didn't see any significant improvement over using a simple float type


# ## All cards info

# In[ ]:


cards_distr = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
cards_probs = {card: Decimal(freq)/Decimal(len(cards_distr))  for card, freq in Counter(cards_distr).items()}


# ## Let's calculate dealer's outcomes probs

# In[ ]:


@lru_cache(maxsize=128)
def calc_probs(value, aces):
    if value > 21:
        if aces > 0 and value <= 31:
            value -= 10
            aces -= 1
        else:
            return {0: Decimal(1)}
    elif value >= 17:
        return {value: Decimal(1)}
    
    res = defaultdict(lambda: 0)
    for card, prob in cards_probs.items():
        for k, v in calc_probs(value + card, aces + int(card == 11)).items():
            res[k] += v * prob
    return res


# In[ ]:


dealer_probs = {x: calc_probs(x, x==11) for x in range(2, 12)}


# ## This functions help us find out the probability to win if the player hits/stays

# In[ ]:


@lru_cache(maxsize=256)
def prob_win_stay(player_total, dealer_card_val):
    dealer_p = dealer_probs[dealer_card_val]
    prob = Decimal(0)
    for res, p in dealer_p.items():
        if res >= player_total:
            prob += p
    return Decimal(1) - prob

@lru_cache(maxsize=1024)
def prob_win_hit(player_total, dealer_card_val, new_card_is_ace=False):
    if player_total > 21:
        if new_card_is_ace and (player_total <= 31):
            player_total -= 10
        else:
            return Decimal(0)
        
    hit_prob = sum([prob_win_hit(player_total + card, dealer_card_val, Decimal(card==11)) * prob
                for card, prob in cards_probs.items()])
    stay_prob = prob_win_stay(player_total, dealer_card_val)
    return max(hit_prob, stay_prob)


# In[ ]:


def should_hit(player_total, dealer_card_val, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """    
    if prob_win_stay(player_total, dealer_card_val) >= prob_win_hit(player_total, dealer_card_val):
        return False
    return True

blackjack.simulate(n_games=1000000)


# **Pretty good result. Yet, we may do even better! Let's find out when we start playing a new game and use this knowledge to improve our chances.**

# In[ ]:


@lru_cache(maxsize=10240)
def prob_win_hit(player_total, dealer_card_val, ace_num):
    if player_total > 21:
        if (ace_num > 0) and (player_total <= 31):
            player_total -= 10
            ace_num -= 1
        else:
            return Decimal(0)
        
    hit_prob = sum([prob_win_hit(player_total + card, dealer_card_val, ace_num + int(card==11)) * prob
                for card, prob in cards_probs.items()])
    stay_prob = prob_win_stay(player_total, dealer_card_val)
    return max(hit_prob, stay_prob)


# In[ ]:


def should_hit(player_total, dealer_card_val, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    new_game = False
    if not should_hit.we_hit:
        new_game = True
    elif dealer_card_val != should_hit.dealer:
        new_game = True
    elif player_aces < should_hit.aces_total:
        new_game = True
    elif (player_total - should_hit.player < -8) and (should_hit.aces_active > 0):
        # Edge case: player received 2 and had an active ace in hand, so 
        # his new score is prev_score + 2 - 10 = prev_score - 8
        new_game = True
    elif (player_total <= should_hit.player) and (should_hit.aces_active == 0):
        # Edge case: player received ace which transforms to 1, so 
        # new_score = prev_score + 1
        new_game = True
    
    if new_game:
        should_hit.aces_active = 1 if player_aces else 0
    else:
        if (player_aces - should_hit.aces_total) == 1: # player got new ace
            should_hit.aces_active += 1
        if player_total - should_hit.player < 2: # player transformed an ace 11->1
            should_hit.aces_active -= 1
    
    should_hit.aces_total = player_aces
    should_hit.player = player_total
    should_hit.dealer = dealer_card_val
    
    if prob_win_stay(player_total, dealer_card_val) >= prob_win_hit(player_total, dealer_card_val, should_hit.aces_active):
        should_hit.we_hit = False
    else:
        should_hit.we_hit = True
    return should_hit.we_hit

should_hit.aces_total = 0
should_hit.aces_active = 0
should_hit.player = 0
should_hit.dealer = 0
should_hit.we_hit = False

blackjack.simulate(n_games=1000000)


# # Discuss Your Results
# 
# How high can you get your win rate? We have a [discussion thread](https://www.kaggle.com/learn-forum/58735#latest-348767) to discuss your results. Or if you think you've done well, reply to our [Challenge tweet](https://twitter.com/kaggle) to let us know.

# ---
# This exercise is from the **[Python Course](https://www.kaggle.com/Learn/python)** on Kaggle Learn.
# 
# Check out **[Kaggle Learn](https://www.kaggle.com/Learn)**  for more instruction and fun exercises.
