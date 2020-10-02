#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Added feature of keeping at least one ship and at least one shipyard active when possible
"""


# Set Up Environment
from kaggle_environments import evaluate, make
env = make("halite", configuration={ "episodeSteps": 400 }, debug=True)
print (env.configuration)


# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', '\nfrom kaggle_environments.envs.halite.helpers import *\nimport random\n\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me = board.current_player\n\n    # at least one shipyard\n    if not me.shipyards:\n        me.ships[0].next_action = ShipAction.CONVERT\n        return me.next_actions\n\n    # at least one ship\n    if not me.ships and me.shipyards:\n        me.shipyards[0].next_action = ShipyardAction.SPAWN\n        return me.next_actions\n\n    # Set ship actions\n    for ship in me.ships:\n        ship.next_action = random.choice(\n            [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST, None])\n\n    # Set shipyard actions\n    for shipyard in me.shipyards:\n        # 10% chance of spawning\n        shipyard.next_action = random.choice([ShipyardAction.SPAWN] + [None] * 9)\n\n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/submission.py", "random","random","random"])
env.render(mode="ipython", width=800, height=600)


# # Submit to Competition
# 1. "Save & Run All" (commit) this Notebook
# 1. Go to the notebook viewer
# 1. Go to "Data" section and find submission.py file.
# 1. Click "Submit to Competition"
# 1. Go to [My Submissions](https://www.kaggle.com/c/halite/submissions) to view your score and episodes being played.
