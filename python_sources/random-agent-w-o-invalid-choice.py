#!/usr/bin/env python
# coding: utf-8

# # Create ConnectX Environment

# In[ ]:


get_ipython().system("pip install 'kaggle-environments>=0.1.4'")


# In[ ]:


import numpy as np

from kaggle_environments import evaluate, make


# In[ ]:


env = make('connectx', debug=True)
env.render()


# # Create my Agent

# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', 'import numpy as np\n\n\ndef my_agent(obs, config):\n    board = np.array(obs.board).reshape(config.rows, config.columns)\n    return int(np.random.choice(np.where(board[0, :] == 0)[0]))')


# In[ ]:


get_ipython().run_line_magic('run', 'submission.py')


# # Test my Agent

# In[ ]:


env.reset()
env.run([my_agent, 'random'])
env.render(mode='ipython', width=400, height=360)


# In[ ]:


env.reset()
env.run([my_agent, my_agent])
env.render(mode='ipython', width=400, height=360)


# # Debug/Train my Agent

# In[ ]:


trainer = env.train([None, "random"])

observation = trainer.reset()

while not env.done:
    my_action = my_agent(observation, env.configuration)
    print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
    env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()


# # Evaluate my Agent

# In[ ]:


def mean_reward(rewards):
    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)

# Run multiple episodes to estimate it's performance.
print("My Agent vs Random Agent:",  mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))
print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))

