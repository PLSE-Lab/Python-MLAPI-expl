#!/usr/bin/env python
# coding: utf-8

# # Install Package

# In[ ]:


get_ipython().system("pip install 'kaggle-environments>=0.1.4'")


# ## Setup Environment

# In[ ]:


from kaggle_environments import evaluate, make
cx_env = make("connectx", debug=True)


# ## Environment Configuration

# In[ ]:


cx_env.configuration


# ## Create Agent

# In[ ]:


def helloworld_agent(observation, configuration):
    # Write your smart logic here...
    import numpy as np
    rows = configuration.rows
    columns = configuration.columns
    current_board = observation.board
    table = np.reshape(current_board,(rows,columns))
    valid_positions = [move for move in range(columns) if observation.board[move] == 0]
    counts = [np.count_nonzero(table[:,move]) for move in valid_positions]
    return valid_positions[np.argmin(counts)]


# # Agent Game
# 
# You're playing with [Negamax](https://en.wikipedia.org/wiki/Negamax) agent which is very smart so you need very hard work to beat him :).

# In[ ]:


cx_env.reset()
cx_env.run([helloworld_agent, 'negamax'])
cx_env.render(mode="ipython", width=650, height=650)


# # Agent Performance

# In[ ]:


def agent_reward(rewards):
    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)

print("CX Agent vs Negamax Agent:", agent_reward(evaluate("connectx", [helloworld_agent, "negamax"], num_episodes=10)))


# # Agent Submission

# In[ ]:


import inspect
import os

def submit_agent(agent, file):
    with open(file, "a" if os.path.exists(file) else "w") as writer:
        writer.write(inspect.getsource(agent))

submit_agent(helloworld_agent, "submission.py")


# ### Please upvote if you find this kernel is useful!
