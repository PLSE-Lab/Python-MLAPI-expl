#!/usr/bin/env python
# coding: utf-8

# # Baseline Idle Bot
# During the first few days of Halite 4, I noticed that a very strong bot in the top 10 was simply building 1 shipyard, leaving 1 ship on his shipyard, and not moving the ship at all. If not attacked, the player would end with 4000 Halite at the end of the game. He won many games, despite other players seemingly controlling the entire board.
# 
# Since then, the effectiveness of such a bot has dropped. The idle bot should still easily rack up 800 points today.
# 
# I think this is an important idea: unless you end the game with lots of Halite, expansion is pointless.
# 
# Make sure your bots are capable of beating this idle bot.

# ## Source Code

# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', 'from kaggle_environments.envs.halite.helpers import *\n\n"""\nCall site.\n"""\ndef agent(observation, configuration):\n    board = Board(observation, configuration)\n    current_player = board.current_player\n\n    # Get my player.\n    if len(current_player.shipyards) == 0:\n        if len(current_player.ships) == 0:\n            return current_player.next_actions\n\n        converting_ship = current_player.ships[0]\n        converting_ship.next_action = ShipAction.CONVERT\n\n    for shipyard in current_player.shipyards:\n        if len(current_player.ships) == 0 and board.current_player.halite >= 500:\n            shipyard.next_action = ShipyardAction.SPAWN\n\n    return current_player.next_actions')


# ## Example Game

# In[ ]:


from kaggle_environments import evaluate, make

env = make("halite", debug=True)


# In[ ]:


env.run(["submission.py", "random", "random", "random"])
env.render(mode="ipython", width=800, height=600)

