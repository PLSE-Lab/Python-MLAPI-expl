#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Set Up Environment
from kaggle_environments import evaluate, make
env = make("halite", configuration={ "episodeSteps": 400 }, debug=True)
print (env.configuration)


# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', '\nfrom kaggle_environments.envs.halite.helpers import *\nfrom random import choice\n\ndef agent(obs,config):\n    \n    board = Board(obs,config)\n    me = board.current_player\n    \n    # Set actions for each ship\n    for ship in me.ships:\n        ship.next_action = choice([ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,None])\n    \n    # Set actions for each shipyard\n    for shipyard in me.shipyards:\n        shipyard.next_action = None\n    \n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/submission.py", "random","random","random"])
env.render(mode="ipython", width=800, height=600)


# # Submit to Competition
# 1. "Save & Run All" (commit) this Notebook
# 1. Go to the notebook viewer
# 1. Go to "Data" section and find submission.py file.
# 1. Click "Submit to Competition"
# 1. Go to [My Submissions](https://www.kaggle.com/c/halite/submissions) to view your score and episodes being played.
