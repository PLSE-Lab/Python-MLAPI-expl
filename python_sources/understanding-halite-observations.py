#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1. Enable Internet in the Kernel (Settings side pane)

# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 
# !curl -X PURGE https://pypi.org/simple/kaggle-environments

# Halite environment was defined in v0.2.1
get_ipython().system("pip install 'kaggle-environments>=0.2.1'")


# # Create Halite environment
# Let's see what the environment looks like, and see a list of available agents

# In[ ]:


from kaggle_environments import evaluate, make

env = make("halite", debug=True)
env.render(mode="ipython", width=800, height=600,controls=True)
print(list(env.agents))


# # The "random" agent
# Let's run this environment on four random agents

# In[ ]:


env.run(["random", "random","random","random"])
env.render(mode="ipython", width=800, height=600)


# Let's create a submission file that logs some useful info from the observations

# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', '\nfrom random import choice\nimport numpy as np\ndef agent(obs):\n    action = {}\n    \n    board=np.reshape(np.float32(obs.halite),(15,15))\n    me=0\n    enemy=1\n    if obs.player==me:\n        current_player="me"\n    else:\n        current_player="enemy"\n    my_total_halite=obs.players[me][0]\n    enemy_total_halite=obs.players[enemy][0]\n    my_shipyards=obs.players[me][1]\n    enemy_shipyards=obs.players[enemy][1]\n    my_ships=obs.players[me][2]\n    enemy_ships=obs.players[enemy][2]\n    print("*"*10)\n    print("At timestep",obs.step)\n    print("Curent player is",obs.player,"This will be important when playing against yourself")\n    print("I have",my_total_halite,"halites")\n    print("Enemy has",enemy_total_halite,"halites")\n    for agent_id in my_shipyards.keys():\n        print("My shipyard",agent_id,"is at",my_shipyards[agent_id],str((my_shipyards[agent_id]%15,my_shipyards[agent_id]//15)))\n    for agent_id in my_ships.keys():\n        print("My ship",agent_id,"is at",my_ships[agent_id][0],str((my_ships[agent_id][0]%15,my_ships[agent_id][0]//15)),"and has",my_ships[agent_id][1],"halite")\n    for agent_id in enemy_shipyards.keys():\n        print("Enemy shipyard",agent_id,"is at",enemy_shipyards[agent_id],str((enemy_shipyards[agent_id]%15,enemy_shipyards[agent_id]//15)))\n    for agent_id in enemy_ships.keys():\n        print("Enemy ship",agent_id,"is at",enemy_ships[agent_id][0],str((enemy_ships[agent_id][0]%15,enemy_ships[agent_id][0]//15)),"and has",enemy_ships[agent_id][1],"halite")\n        \n    ship_action=choice(["NORTH", "SOUTH", "EAST", "WEST", None])\n    ship_id=choice(list(my_ships.keys())+list(my_shipyards.keys()))\n    if ship_action is not None:\n        action[ship_id] = ship_action\n    print("Action taken:",action)\n    return action')


# Let's run our random agent against the default one. Co-ordinates (0,0) are at the top left corner.

# In[ ]:


# Play as the first agent against default "shortest" agent.
env.run(["/kaggle/working/submission.py", "random"])
env.render(mode="ipython", width=800, height=600)


# In[ ]:




