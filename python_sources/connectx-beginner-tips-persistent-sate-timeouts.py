#!/usr/bin/env python
# coding: utf-8

# # ConnectX Beginner Tips: Persistent Sate & Timeouts
# 
# Some of the info in the starter notebooks is out-of-date, so I thougt I'd share some tips that will make it easier to get started. These are all things that tripped me up in the beginning.
# 
# 1. **Your agent functions don't have to be standalone.** You can write a python file with definitions at the top level as you normally would. Just make sure your agent function is the last definition in the file.
# 
# 
# 2. **You can use global variables for persistent state.** This lets you store data across turns. For example, if you want to try Q-learning, you can save your value estimates and improve them as the game progresses.
# 
# 
# 3. **Use `config.actTimeout` to get as much work done during your turn as possible.** You can also use config.timeout, which is the older name. They are the same (at least for now).
# 
# 
# Any additional tips are appreciated!
# 
# 
# The rest of this notebook illustrates a simple agent that keeps track of its turn number to drop its pieces from left to right.

# In[ ]:


from kaggle_environments import make

env = make("connectx", debug=True)


# The next cell contains the code for our agent. We write out the whole cell to "submission.py" using the `%%writefile` magic. Be sure to comment out `%%writefile` when you're working, otherwise executing the cell doesn't actually evaluate the code.

# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', '\n# We can import where we normally would.\nimport time\n\n# We can have global variables to store persistent state across turns.\nturn = -1\n\n# We can define our helper functions at the top level, which makes them easier to debug.\ndef increment(turn, config):\n    return (turn + 1) % config.columns\n\n\n# Our agent. This function needs to be the last one in the file, \n# so keep it at the bottom of this cell. \ndef agent_sequential(obs, config):\n    global turn\n    \n    # Set a deadline so we return before the timeout. Be sure to include a buffer \n    # so you never accidently go over!\n    deadline = time.time() + config.actTimeout - 0.5\n    \n    while time.time() < deadline:\n        # Do lots of very important calculuations\n        pass\n    \n    # Just drop our next piece in column turn % 7. This will go from left to right\n    # across the board.\n    turn = increment(turn, config)\n        \n    return turn')


# In[ ]:


# Validate submission file. This code is from the "Intro to Game AI and Reinforcement Learning" course.
# https://www.kaggle.com/alexisbcook/play-the-game

import sys
from kaggle_environments import utils

out = sys.stdout
submission = utils.read_file("/kaggle/working/submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, "random"])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")


# The next cell loads up our agent since we never actually evaluated the code in cell [2].

# In[ ]:


get_ipython().run_line_magic('run', 'submission.py')


# In[ ]:


# Two agents play one game round
env.run([agent_sequential, "random"]);
# Show the game
env.render(mode="ipython")

