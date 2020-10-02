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

# In[ ]:


def should_hit(player_total, dealer_card_val, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    sh = 0 if (dealer_card_val < 8) else 1
    hit = False
    if player_total < 18:
        if player_total>12+sh and player_aces:
            hit = True
        elif player_total < 13+sh:        
            hit=True
    return hit

blackjack.simulate(n_games=50000)


# # Discuss Your Results
# 
# How high can you get your win rate? We have a [discussion thread](https://www.kaggle.com/learn-forum/58735#latest-348767) to discuss your results. Or if you think you've done well, reply to our [Challenge tweet](https://twitter.com/kaggle) to let us know.

# ---
# This exercise is from the **[Python Course](https://www.kaggle.com/Learn/python)** on Kaggle Learn.
# 
# Check out **[Kaggle Learn](https://www.kaggle.com/Learn)**  for more instruction and fun exercises.
