#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !curl -X PURGE https://pypi.org/simple/kaggle-environments
get_ipython().system("pip install 'kaggle-environments>=0.2.1'")


# This is my collection of simple bad agents for Halite.
# 
# These are all very simple agents that control at most one ship.  I'm certain they won't win anything.  Building them helped me think about how the game works.  The agents are:
# 
# **yard_only_agent** - builds a shipyard and then does nothing else
# 
# **runaway_agent** - runs away from other ships
# 
# **run_yard_agent** - runs away from others until it has enough cargo to build a shipyard
# 
# **always_one_agent** - builds a shipyard on the first turn then builds a ship if one doesn't exist
# 
# **run_yard_one_agent** - combination of run_yard and always building a replacement ship
# 
# **take_invalid_action_agent** - confirms that your game ends after taking an invalid action
# 

# Before we get to the part about building agents, we'll need some way to tell the score after the game is over.  I'm not going to calculate how to break ties, I'll just print the score at the end of the game.

# In[ ]:


def display_score(env):
    for state in env.steps[-1]:
        player = state.observation.player
        score = state.reward
        print(f'Player {player}: {score}')


# ### yard_only_agent
# Ok, simplest one first.  It will create a shipyard on the first turn and then...nothing.  This may sound useless, but consider that this agent:
# * always survives until the end of the game
# * always scores 3000 points
# * consistently beats the builtin "random" agent

# In[ ]:


# I wanted an agent that would live the entire game
# The "random" agent doesn't always do that
def yard_only_agent(obs, config):
    actions = {}
    me = obs.players[obs.player]
    num_yards = len(me[1])
    if num_yards == 0:
        first_ship = next(iter(me[2].keys()))
        actions = { first_ship: "CONVERT" }
    return actions


# ### runaway_agent
# 
# This agent controls a single ship that runs away from any other ships and avoids enemy shipyards.  Since I'm going to use this agent as a basis for other agents it has the additional ability that it will always move to an adjacent square with more halite.  This agent:
# * almost always survives until the end of the game
# * if it survives it scores 5000 (beating yard_only_agent)
# * it collects halite (but it doesn't do anything with it!)

# In[ ]:


import random 

from kaggle_environments.envs.halite.halite import get_to_pos

def create_enemy_ship_possible_map(obs, config):
    map = [0] * config.size * config.size
    player = obs.player
    for idx, opp in enumerate(obs.players):
        if idx == player:
            continue
        for ship in opp[2].values():
            map[ship[0]] = 1
            for dir in ["NORTH", "SOUTH", "EAST", "WEST"]:
                map[get_to_pos(config.size, ship[0], dir)] = 1
    return map

def create_enemy_yard_map(obs, config):
    map = [0] * config.size * config.size
    player = obs.player
    for idx, opp in enumerate(obs.players):
        if idx == player:
            continue
        for yard_pos in opp[1].values():
            map[yard_pos] = 1
    return map

def runaway_agent(obs, config):
    me = obs.players[obs.player]
    first_ship = next(iter(me[2].keys()))
    pos = me[2][first_ship][0]
    cargo = me[2][first_ship][1]
    esp = create_enemy_ship_possible_map(obs, config)
    ey = create_enemy_yard_map(obs, config)
    bad_square = [a+b for a,b in zip(esp, ey)]
    # ignore negative halite
    good_square = [x if x >= 0 else 0 for x in obs.halite]
    square_score = [-b if b > 0 else g for b,g in zip(bad_square, good_square)]
    moves = ["NORTH", "SOUTH", "EAST", "WEST"]
    random.shuffle(moves)
    best_score = square_score[pos]
    best_move = ""
    actions = {}
    for move in moves:
        new_pos = get_to_pos(config.size, pos, move)
        pos_score = square_score[new_pos]
        if pos_score > best_score or (pos_score == 0 and best_score == 0):
            best_score = pos_score
            best_move = move
            actions = { first_ship: best_move }
    return actions


# #### run_yard_agent
# 
# This agent has a single ship that wanders the board like runaway_agent until it has collected enough halite to CONVERT.  This agent:
# * always (assuming survival) scores more than runaway_agent because you are credited with any excess halite when you CONVERT
# 

# In[ ]:


def run_yard_agent(obs, config):
    actions = {}
    me = obs.players[obs.player]
    if len(me[2]) > 0:  # if I have a ship
        first_ship = next(iter(me[2].keys()))
        if me[2][first_ship][1] > config.convertCost:
            actions = { first_ship: "CONVERT" }
        else:
            actions = runaway_agent(obs, config)
    return actions


# #### always_one_agent
# 
# This agent builds a shipyard on it's first turn and then always builds one ship if there isn't one.  This agent:
# * almost always scores 2500
# * isn't that good, because the ship it builds acts like runaway_agent (which doesn't know how to deposit halite!)

# In[ ]:


def always_one_agent(obs, config):
    me = obs.players[obs.player]
    num_yards = len(me[1])
    num_ships = len(me[2])
    if num_yards == 0:
        first_ship = next(iter(me[2].keys()))
        actions = { first_ship: "CONVERT" }
    elif num_ships == 0:
        first_yard = next(iter(me[1].keys()))
        actions = { first_yard: "SPAWN" }
    else:
        actions = runaway_agent(obs, config)
    return actions


# #### run_yard_one_agent
# 
# This agent controls one ship that wanders the board until it has collected enough cargo to CONVERT.  When a shipyard is created it builds a new ship that does the same thing.  This agent:
# * builds multiple shipyards (but only 1 ship at time!)
# * scores incredibly well against bad opponents
# * against even half-way decent opponents would not do nearly as well

# In[ ]:


def run_yard_one_agent(obs, config):
    me = obs.players[obs.player]
    num_ships = len(me[2])
    if num_ships == 0:
        first_yard = next(iter(me[1].keys()))
        actions = { first_yard: "SPAWN" }
    else:
        actions = run_yard_agent(obs, config)
    return actions


# #### take_invalid_action_agent
# 
# I built this agent to convince myself that I understood what happened when you took an invalid action.  What happens when you take an invalid action is that your player is disqualified and you get no more moves for the remainder of the game.

# In[ ]:


def take_invalid_action_agent(obs, config):
    print("Action requested")
    return {"Non-existant ship": "CONVERT"}


# Here's a few of my bad agents in action.

# In[ ]:


from kaggle_environments import make

env = make("halite", configuration={"agentExec": "LOCAL"}, debug=True)
env.run(["random", yard_only_agent, runaway_agent, take_invalid_action_agent])
env.render(mode="ipython", width=800, height=600)
display_score(env)


# In[ ]:


env = make("halite", configuration={"agentExec": "LOCAL"}, debug=True)
env.run(["random", run_yard_agent, always_one_agent, run_yard_one_agent])
env.render(mode="ipython", width=800, height=600)
display_score(env)


# #### Improvements
# 
# Let's attempt to put some of these ideas together into a reasonable agent.  Here are the rules for the new agent:
# 
# Shipyards:
# * spawn new ships until we have at least half as many ships as the biggest opponent
# 
# Ships:
# * wait to unload cargo if at a friendly shipyard
# * wait until the current square is empty if we have cargo and there is no danger 
# * build a shipyard if we have no shipyards and some cargo
# * build a shipyard if we have enough cargo to convert and spawn
# * build a shipyard if we have enough cargo and it is the last move
# * move to an adjacent space with better halite and no danger
# 

# In[ ]:


from enum import Enum
from collections import OrderedDict

def create_friendly_ship_possible_map(obs, config, ignore_id):
    map = [0] * config.size * config.size
    player = obs.player
    me = obs.players[player]
    for id, ship in me[2].items():
        if id == ignore_id:
            continue
        map[ship[0]] = 1
        for dir in ["NORTH", "SOUTH", "EAST", "WEST"]:
            map[get_to_pos(config.size, ship[0], dir)] = 1
    return map

def create_friendly_yard_map(obs, config):
    map = [0] * config.size * config.size
    player = obs.player
    me = obs.players[player]
    for yard_pos in me[1].values():
        map[yard_pos] = 1
    return map

def yard_actions(obs, config):
    actions = {}
    me = obs.players[obs.player]
    bank = me[0]
    num_yards = len(me[1])
    num_ships = len(me[2])
    max_ships = max([len(p[2]) for p in obs.players])
    if (num_ships < (max_ships/2) or num_ships == 0) and bank > config.spawnCost:
        pick_yard = random.choice([id for id in me[1].keys()])
        actions[pick_yard] = "SPAWN"
    return actions

def ship_actions(obs, config):
    actions = {}
    is_last_move = (obs.step == config.episodeSteps-2)
    moves = ["NORTH", "SOUTH", "EAST", "WEST"]
    esp = create_enemy_ship_possible_map(obs, config)
    ey = create_enemy_yard_map(obs, config)
    fy = create_friendly_yard_map(obs, config)
    good_square = [x if x >= 0 else 0 for x in obs.halite]
    me = obs.players[obs.player]
    num_yards = len(me[1])
    for id in me[2]:
        pos = me[2][id][0]
        cargo = me[2][id][1]
        fsp = create_friendly_ship_possible_map(obs, config, id)
        bad_square = [a+b+c for a,b,c in zip(esp, ey, fsp)]
        square_score = [-b if b > 0 else g for b,g in zip(bad_square, good_square)]
        if cargo > 0 and square_score[pos] > 0:
            # stay on a positive square and collect if we already have cargo
            pass
        elif cargo > 0 and fy[pos] > 0:
            # stay on shipyard to unload cargo
            pass
        elif cargo > (config.convertCost + config.spawnCost):
            # we have enough cargo to build a yard and replace this ship
            actions[id] = "CONVERT"
        elif cargo > 0 and num_yards == 0:
            # we've completely mined (see above) our first square, build our first shipyard
            actions[id] = "CONVERT"
        elif cargo > config.convertCost and is_last_move:
            # this is the last move and this ship isn't at a shipyard (see above)
            actions[id] = "CONVERT"
        else:
            # move to better adjacent halite while avoiding danger
            best_score = square_score[pos]
            random.shuffle(moves)
            for move in moves:
                new_pos = get_to_pos(config.size, pos, move)
                pos_score = square_score[new_pos]
                if pos_score > best_score or (pos_score == 0 and best_score == 0):
                    best_score = pos_score
                    actions[id] = move
    return actions

class Actions(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4
    CONVERT = 5
    SPAWN = 6

def simple_agent(obs, config):
    actions = {}
    me = obs.players[obs.player]
    if len(me[1]) > 0:
        actions.update(yard_actions(obs, config))
    if len(me[2]) > 0:
        actions.update(ship_actions(obs, config))
    # due to a kaggle bug all actions need to be ordered so that CONVERT/SPAWN are last
    actions_list = list(actions.items())
    sorted_list = sorted(actions_list, key=lambda x: Actions[x[1]].value)
    # I realize regular dicts are order preserving in python 3.7 and
    # also in 3.6 as an implementation detail.  I didn't want to make
    # any assumptions about the python version kaggle uses
    actions = OrderedDict(sorted_list)
    return actions


# This agent easily beats my bad agents on almost all maps.  Against an agent that is smart about collecting and depositing halite in the shipyards this agent wouldn't fare nearly as well.

# In[ ]:


env = make("halite", configuration={"agentExec": "LOCAL"}, debug=True)
env.run(["random", run_yard_agent, run_yard_one_agent, simple_agent])
env.render(mode="ipython", width=800, height=600)
display_score(env)


# #### Additional Improvements
# 
# There are, of course, some incredibly obvious improvements that could be made.  For instance:
# * tell the ships how and when to return to the shipyard to deposit cargo
# * tell the ships how to go directly to big halite squares instead of wandering around until they are adjacent to them
# * coordinate among friendly ships so that the ships aren't always running away from one another
# * stop building shipyards next to one another
# 
# But these sound useful enough to move these agents out of the "bad agents" category so I'll stop this notebook now.
# 

# In[ ]:




