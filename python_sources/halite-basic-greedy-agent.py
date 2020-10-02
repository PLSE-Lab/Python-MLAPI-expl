#!/usr/bin/env python
# coding: utf-8

# # Basic Greedy Agent
# 
# A simple agent, created for gaining understanding of the observation. It may be useful to train future agents against. Constants are hardcoded and probably suboptimal.
# 
# Maintains one shipyard and one ship.

# In[ ]:


get_ipython().system("pip install 'kaggle-environments>=0.2.1'")

import kaggle_environments
print("Kaggle Environments version:", kaggle_environments.version)


# In[ ]:


get_ipython().run_cell_magic('writefile', 'agent.py', '\nDIRS = ["NORTH", "SOUTH", "EAST", "WEST"]\n\n# Each ship id will be assigned a state, one of COLLECT or DEPOSIT, which decides what it will do on a turn.\nstates = {}\n\nCOLLECT = "collect"\nDEPOSIT = "deposit"\n\n\ndef argmax(arr, key=None):\n  return arr.index(max(arr, key=key)) if key else arr.index(max(arr))\n\n\n# This function will not hold up in practice\n# E.g. cell getAdjacent(224) includes position 0, which is not adjacent\ndef getAdjacent(pos):\n  return [\n    (pos - 15) % 225,\n    (pos + 15) % 225,\n    (pos +  1) % 225,\n    (pos -  1) % 225\n  ]\n\ndef getDirTo(fromPos, toPos):\n  fromY, fromX = divmod(fromPos, 15)\n  toY,   toX   = divmod(toPos,   15)\n\n  if fromY < toY: return "SOUTH"\n  if fromY > toY: return "NORTH"\n  if fromX < toX: return "EAST"\n  if fromX > toX: return "WEST"\n\n    \ndef agent(obs):\n  action = {}\n\n  player_halite, shipyards, ships = obs.players[obs.player]\n\n  for uid, shipyard in shipyards.items():\n    # Maintain one ship always\n    if len(ships) == 0:\n      action[uid] = "SPAWN"\n\n  for uid, ship in ships.items():\n    # Maintain one shipyard always\n    if len(shipyards) == 0:\n      action[uid] = "CONVERT"\n      continue\n\n    # If a ship was just made\n    if uid not in states: states[uid] = COLLECT\n\n    pos, halite = ship\n\n    if states[uid] == COLLECT:\n      if halite > 2500:\n        states[uid] = DEPOSIT\n\n      elif obs.halite[pos] < 100:\n        best = argmax(getAdjacent(pos), key=obs.halite.__getitem__)\n        action[uid] = DIRS[best]\n\n    if states[uid] == DEPOSIT:\n      if halite < 200: states[uid] = COLLECT\n\n      direction = getDirTo(pos, list(shipyards.values())[0])\n      if direction: action[uid] = direction\n      else: states[uid] = COLLECT\n\n\n  return action')


# # View the agent at work

# In[ ]:


# Sparring Partner
def null_agent(*_): return {}

for _ in range(3):
    env = kaggle_environments.make("halite", debug=True)
    env.run(["agent.py", null_agent])
    env.render(mode="ipython", width=800, height=600)


# # Evaluate Agent
# Extended from [Halite Getting Started](https://www.kaggle.com/ajeffries/halite-getting-started)

# In[ ]:


def mean_reward(rewards):
    wins = 0
    ties = 0
    loses = 0

    for r in rewards:
        r0 = r[0] or 0
        r1 = r[1] or 0

        if   r0 > r1: wins  += 1
        elif r1 > r0: loses += 1
        else:         ties  += 1

    return [wins / len(rewards), ties / len(rewards), loses / len(rewards)]


import inspect
def test_against(enemy, n=25):
    results = mean_reward(kaggle_environments.evaluate(
        "halite",
        ["agent.py", enemy],
        num_episodes=n
    ))

    enemy_name = enemy.__name__ if inspect.isfunction(enemy) else enemy
    print("My Agent vs {}: wins={}, ties={}, loses={}".format(enemy_name, *results))

test_against(null_agent)
test_against("random")

