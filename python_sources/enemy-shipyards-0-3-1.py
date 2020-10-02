#!/usr/bin/env python
# coding: utf-8

# # The bug displayed in this notebook has been fixed in 0.3.2

# In[ ]:


get_ipython().system('pip install kaggle-environments==0.3.1')


# Here are two simple agents.  One agent builds a shipyard, spawns a ship and then moves east.  The other agent builds a shipyard, spawns a ship and then moves south.

# In[ ]:


def east_agent(obs, config):
    actions = {}
    me = obs.players[obs.player]
    num_yards = len(me[1])
    num_ships = len(me[2])
    for ship_id in me[2].keys():
        actions[ship_id] = "EAST"
    if num_yards == 0:
        first_ship = next(iter(me[2].keys()))
        actions[first_ship] = "CONVERT"
    elif num_ships == 0:
        first_yard = next(iter(me[1].keys()))
        actions[first_yard] = "SPAWN"
    return actions


def south_agent(obs, config):
    actions = {}
    me = obs.players[obs.player]
    num_yards = len(me[1])
    num_ships = len(me[2])
    for ship_id in me[2].keys():
        actions[ship_id] = "SOUTH"
    if num_yards == 0:
        first_ship = next(iter(me[2].keys()))
        actions[first_ship] = "CONVERT"
    elif num_ships == 0:
        first_yard = next(iter(me[1].keys()))
        actions[first_yard] = "SPAWN"
    return actions
    


# In[ ]:


from kaggle_environments import make

env = make("halite")
env.run([east_agent, south_agent])
env.render(mode="ipython", width=800, height=600)


# Why does this game end at step 14?
# 
# At step 13 the yellow ship crashes into the red shipyard, but nothing happens.
# 
# On step 14 a collision is drawn to the east of the red shipyard and the yellow agent is eliminated.  
# 
# The yellow agent still had a shipyard and enough halite to build another ship.  Why was that agent eliminated?

# In[ ]:




