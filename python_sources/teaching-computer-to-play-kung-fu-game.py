#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().system('pip install h5py')


# In[ ]:


get_ipython().system('pip install gym')


# In[ ]:


get_ipython().system('pip install gym[atari]')


# In[ ]:


get_ipython().system('pip install keras-rl')


# In[ ]:


import gym

env = gym.make('KungFuMaster-ram-v0')


# In[ ]:


actions = env.action_space.n


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam


# In[ ]:


model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape ))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(actions))
model.add(Activation('linear'))
print(model.summary())


# In[ ]:


from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


# In[ ]:


policy = EpsGreedyQPolicy()


# In[ ]:


memory = SequentialMemory(limit=50000, window_length=1)


# In[ ]:


dqn = DQNAgent( model = model, nb_actions = actions, 
memory = memory, nb_steps_warmup=10, policy=policy)


# In[ ]:


dqn.compile(Adam(lr=1e-3), metrics=['mae'])


# In[ ]:


dqn.fit(env, nb_steps=1000, visualize=True, verbose=2)

# set visualize = False if you don't want to see the progress of the agent


# In[ ]:


dqn.fit(env, nb_steps=1000, visualize=True, verbose=2)

# set visualize = False if you don't want to see the progress of the agent


# In[ ]:


import tensorflow as tf
print(tf.__version__)


# In[ ]:


get_ipython().system('pip install tensorflow==1.14.0')

