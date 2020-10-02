#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install kaggle-environments')


# In[ ]:


from kaggle_environments import make
from IPython.core.display import HTML

env = make("connectx")
print(env.name, env.version)
print("Default Agents: ", *env.agents)


# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', "import numpy as np\nplayer_one = True\n\ndef find_winnig_moves(board):\n    moves = []\n    return moves\n\ndef find_defending_moves(board):\n    moves = []\n    return moves\n\ndef agent(observation, configuration):\n    global player_one\n    moves = []\n    board = observation.board\n    board = np.array(board).reshape(configuration.rows,configuration.columns)\n    \n    if np.sum(board) == 1: player_one = False\n    if np.sum(board) < 2: print('Player 1:', player_one)\n    \n    if np.sum(board) < 7: #Random Start\n        moves += [int(np.random.choice(np.arange(configuration.columns).astype(int)))]\n    else:\n        moves = find_winnig_moves(board)\n        moves += find_defending_moves(board)\n\n    for i in range(configuration.columns): #Add Any Open\n        if np.sum([1 for m in board[:,i] if m ==0])>0:\n            moves +=[i]\n    return moves[0]")


# In[ ]:


get_ipython().run_line_magic('run', 'submission.py')


# In[ ]:


env = make("connectx", debug=True)
env.run([agent, "random"])
HTML(env.render(mode="ipython", width=600, height=500, header=False))


# In[ ]:


env.run(["random", agent])
HTML(env.render(mode="ipython", width=600, height=500, header=False))

