#!/usr/bin/env python
# coding: utf-8

# Upload the replay to you input data
# https://www.kaggle.com/dansbecker/finding-your-files-in-kaggle-kernels

# In[ ]:


import json
with open('../input/1209159.json') as f:
    replay = json.load(f)


# In[ ]:


def agent(obs):
    return {}


# In[ ]:


from kaggle_environments import evaluate, make
#replay['configuration']['randomSeed'] = 1209159
env = make("halite", debug=True, configuration=replay['configuration'])
_ = env.reset(num_agents=4)


# In[ ]:


TURNS_TO_REPLAY = 100
MY_PLAYER = 3
current_turn = 1
while current_turn < TURNS_TO_REPLAY:
    observation = replay['steps'][current_turn][0]['observation'].copy()
    observation['player'] = MY_PLAYER
    _ = agent(observation)
    _ = env.step([replay['steps'][current_turn][player]['action'] for player in range(4)])
    current_turn += 1


# In[ ]:


env.render(mode="ipython", header=True, controls=True, width=800, height=600)

