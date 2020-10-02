#!/usr/bin/env python
# coding: utf-8

# # [This notebook became obsolete with the 0.3 release of kaggle_environments.]
# 
# # HaliteHack
# 
# HaliteHack is an updated environment for playing halite.  It includes UI improvements and a simple logging capability.  Please submit any bug reports or improvements to my [halitehack github](https://github.com/hubcity/halitehack).
# 
# ## UI Improvements
# 
# HaliteHack makes some very basic UI improvements to the ipython/html renderer used by Halite.
# - Score for each player is shown
# - Empty ships are denoted by a black mark in the center of the ship
# - Ships with cargo are denoted by a light blue (halite-colored) mark in the center of the ship
# - Negative halite is no longer drawn on the map
# - Fix: INFO messages are reset on every step [v0.2]
# 
# (These changes look fine to me on a 800x600 map.  Changing the map size to something else will give unpredictable results.)
# 

# In[ ]:


get_ipython().system("pip install 'kaggle-environments==0.2.1'")


# For this kaggle notebook halitehack has been install in /kaggle/input/halitehack/halitehack-v0.2.

# In[ ]:


import sys
sys.path.append('/kaggle/input/halitehack/halitehack-v0.2')


# In[ ]:


import halitehack
print(halitehack.halitehack.specification['name'])


# Yep, it looks like it worked.  To get the new UI ask for a "halitehack" environment instead of a "halite" environment.  That's it.

# In[ ]:


import random
from kaggle_environments import make

random.seed(7)
env = make("halitehack", configuration={"agentExec": "LOCAL"})

env.run(["random", "random", "random", "random"])
env.render(mode="ipython", width=800, height=600)


# The random agent is quite bad unless it finds itself on well-suited map.  That seems to happen very rarely.  In this game which ended with a tremendous amount of halite on the board the top score was 25.

# ## Simple Logging
# 
# The "halitehack" environment adds a new action called INFO.  With this action you can log short informational strings to the UI.  Maybe it's best to show an example.  This code adds the total cargo to the UI for the random agents.

# In[ ]:


from kaggle_environments.envs.halite.halite import random_agent

def random_cargo_info(obs, config):
    actions = {}
    # compute total cargo
    total_cargo = 0.0
    for _, cargo in obs.players[obs.player][2].values():
        total_cargo += cargo
    # add our INFO logging of total cargo
    actions[f"Total cargo: {total_cargo}"] = "INFO"
    # add the actions to be taken by random agent
    actions.update(random_agent(obs, config))
    return actions

random.seed(7)
env = make("halitehack", configuration={"agentExec": "LOCAL"})

env.run([random_cargo_info, random_cargo_info, random_cargo_info, random_cargo_info])
env.render(mode="ipython", width=800, height=600)


# Yes, this new "feature" is particularly ugly.  You should only give one INFO action.  And you have to give it kind of backward - the message that you want to appear on the UI should be the key and the word INFO should be the value.  It's useful if you want to track one piece of information without digging though log files.
# 
# My game showed that the random agent occasionally collects a lot of halite.  The problem is that the halite nevers gets deposited.  Let's add a rule to make the situation a little better.  Deciding when to go to a shipyard and finding the way there would take more code than I'm willing to write.  Let's add simple rule that will tell a ship to CONVERT anytime converting would be cheaper than moving.

# In[ ]:


def random_with_convert(obs, config):
    actions = {}
    # compute total cargo
    total_cargo = 0.0
    for _, cargo in obs.players[obs.player][2].values():
        total_cargo += cargo
    # add our INFO logging of total cargo
    actions[f"Total cargo: {total_cargo}"] = "INFO"
    # add the actions to be taken by random agent
    actions.update(random_agent(obs, config))
    # add CONVERTs if helpful
    for ship_id, (pos, cargo) in obs.players[obs.player][2].items():
        if config.convertCost < cargo * config.moveCost:
            actions[ship_id] = "CONVERT"
    return actions

random.seed(7)
env = make("halitehack", configuration={"agentExec": "LOCAL"})

env.run([random_with_convert, random_with_convert, random_with_convert, random_with_convert])
env.render(mode="ipython", width=800, height=600)


# The rule doesn't help the green agent until step 301 or the yellow agent until step 350.  But it does make the final scores look better.  Now when the random agent lives to the end of the game it might put up a score that is more appropriate for the amount of halite that was collected.
# 
# ## Caveats
# 
# **If you use this for development be sure to remove any INFO actions before submitting your agent to kaggle.  While the "halitehack" environment will happily accept INFO actions, the "halite" environment will not!** 
# 
# Also, the INFO message should never exactly match a unit's uid.  You will overwrite the unit's previously selected action.
# 
# ## Implementation Details
# 
# This code uses the previously unused info field in the agent's dictionary which is stored in the state array.  The halitehack interpreter simply strips all of the INFO actions out, stores them in the info field of the agent's dictionary and then sends the remaining actions to the real halite interpreter.  What all this means is that as long as the next halite release doesn't add an INFO action or start using the info field then this same version of halitehack will continue to work just fine.

# ## Conclusion (Don't use this, upgrade kaggle_environments to the latest version.)
# 
# Here is a replay of my best agent playing in single player mode.  The fun starts around time step 195.  Ships go out to collect the last of the non-negative halite.  At step 212 an INFO message notes that all halite was mined.  The ships with cargo (and a slightly blue center) return to shipyards to depoit cargo.  Then at step 221 an INFO message says that all halite has been delivered.

# In[ ]:


import json 

with open('/kaggle/input/halitehack/replays/complete.json') as json_file:
    steps = json.load(json_file)
    
env = make("halitehack", configuration={"agentExec": "LOCAL"}, steps=steps)
env.render(mode="ipython", width=800, height=600)


# I hope that kaggle will come out with a new renderer soon.  As clunky as this is, it is at the limits of my UI skills.  Please share any improvements that you make.

# In[ ]:




