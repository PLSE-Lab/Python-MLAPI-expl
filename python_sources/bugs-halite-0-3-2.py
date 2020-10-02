#!/usr/bin/env python
# coding: utf-8

# # The bugs described in this notebook have been fixed in kaggle-environments 0.3.9

# In[ ]:


get_ipython().system('pip install kaggle-environments==0.3.2')


# # Occupied Spawn
# 
# Ships can be successfully spawned while a shipyard is occupied.  Based on the text of the rules this should result in a collision.

# In[ ]:


def east_agent(obs, config):
    actions = {}
    me = obs.players[obs.player]
    num_yards = len(me[1])
    num_ships = len(me[2])
    for i, ship_id in enumerate(me[2].keys()):
        if obs.step != 2:
            if i % 2 == 0:
                actions[ship_id] = "EAST"
            else:
                actions[ship_id] = "WEST"
    if num_yards == 0:
        first_ship = next(iter(me[2].keys()))
        actions[first_ship] = "CONVERT"
    elif num_ships == 0 or obs.step == 2:
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
    


# On step 4 a ship is spawned while an empty ship is in the shipyard.  You see an explosion.  Both ships should be destroyed.  On the next step two ships emerge from the shipyard.

# In[ ]:


from kaggle_environments import make

env = make("halite", configuration={"episodeSteps": 10})
env.run([east_agent, south_agent])
env.render(mode="ipython", width=800, height=600)


# When a ship with cargo is in the shipyard during a spawn it should be destroyed, but the new ship should live.

# In[ ]:


def east_agent2(obs, config):
    actions = {}
    me = obs.players[obs.player]
    num_yards = len(me[1])
    num_ships = len(me[2])
    for i, ship_id in enumerate(me[2].keys()):
        if obs.step == 2:
            actions[ship_id] = "WEST"
        elif obs.step == 3:
            pass
        elif obs.step == 4:
            actions[ship_id] = "EAST"
        elif obs.step > 4:
            if i % 2 == 0:
                actions[ship_id] = "EAST"
            else:
                actions[ship_id] = "WEST"
    if num_yards == 0:
        first_ship = next(iter(me[2].keys()))
        actions[first_ship] = "CONVERT"
    elif num_ships == 0 or obs.step == 4:
        first_yard = next(iter(me[1].keys()))
        actions[first_yard] = "SPAWN"
    return actions


# After step 5 a ship with cargo will go to the shipyard while a new ship is being spawned.  Again, both ships survive.

# In[ ]:


import random
import numpy as np

from kaggle_environments import make

random.seed(0)
np.random.seed(1)
env = make("halite", configuration={"episodeSteps": 10})
env.run([east_agent2, south_agent])
env.render(mode="ipython", width=800, height=600)


# # Simultaneous Shipyard Collision
# 
# This bug occurs when multiple ships including at least one enemy ship are in a shipyard at the same time.  I'll demonstrate two different ways this can occur.
# 
# ## Enemy Collision
# When two or more ships collide with the same enemy shipyard the halite game crashes.

# In[ ]:


def east_agent3(obs, config):
    actions = {}
    me = obs.players[obs.player]
    num_yards = len(me[1])
    num_ships = len(me[2])
    for i, ship_id in enumerate(me[2].keys()):
        if i % 2 == 1:
            actions[ship_id] = "EAST"
        else:
            actions[ship_id] = "WEST"
    if num_yards == 0:
        first_ship = next(iter(me[2].keys()))
        actions[first_ship] = "CONVERT"
    elif num_ships == 0 or obs.step == 2:
        first_yard = next(iter(me[1].keys()))
        actions[first_yard] = "SPAWN"
    return actions


# Up until step 13 everything is fine.

# In[ ]:


from kaggle_environments import make

# SET episodeSteps to 13
env = make("halite", configuration={"episodeSteps": 13})
env.run([east_agent3, south_agent])
env.render(mode="ipython", width=800, height=600)


# But run the game to step 14 where two ships collide with an enemy shipyard and the game crashes.

# In[ ]:


from kaggle_environments import make

# CHANGED episodeSteps to 14
env = make("halite", configuration={"episodeSteps": 14})
env.run([east_agent3, south_agent])
env.render(mode="ipython", width=800, height=600)


# ## Occupied Collision
# 
# In this case an enemy ship collides with an occupied shipyard.

# In[ ]:


def east_agent3b(obs, config):
    actions = {}
    me = obs.players[obs.player]
    num_yards = len(me[1])
    num_ships = len(me[2])
    for i, ship_id in enumerate(me[2].keys()):
        actions[ship_id] = "EAST"
    if num_yards == 0:
        first_ship = next(iter(me[2].keys()))
        actions[first_ship] = "CONVERT"
    elif num_ships == 0:
        first_yard = next(iter(me[1].keys()))
        actions[first_yard] = "SPAWN"
    return actions


def occupy_agent(obs, config):
    actions = {}
    me = obs.players[obs.player]
    num_yards = len(me[1])
    num_ships = len(me[2])
    if obs.step == 0:
        first_ship = next(iter(me[2].keys()))
        actions[first_ship] = "CONVERT"
    elif obs.step == 1:
        first_yard = next(iter(me[1].keys()))
        actions[first_yard] = "SPAWN"
    return actions


# After step 12 a ship collides with an occupied enemy shipyard.  The enemy ship survives the collision.  Shouldn't the enemy ship be destroyed?

# In[ ]:


import random
import numpy as np

from kaggle_environments import make

random.seed(0)
np.random.seed(1)

env = make("halite", configuration={"episodeSteps": 15})
env.run([east_agent3b, occupy_agent])
env.render(mode="ipython", width=800, height=600)


# # Negative Halite
# 
# The new halite seeding algorithm sometimes results in negative halite.

# In[ ]:


import random
import numpy as np


def echo_min_halite_agent(obs, config):
    actions = {}
    print(f"Step {obs.step} min halite square {min(obs.halite)}")
    return actions

random.seed(0)
np.random.seed(0)
env = make("halite", configuration={"episodeSteps": 15})
env.run([echo_min_halite_agent, "random", "random", "random"])
env.render(mode="ipython", width=800, height=600)


# In[ ]:




