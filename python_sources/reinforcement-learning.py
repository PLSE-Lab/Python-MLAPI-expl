#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow==1.14')


# In[ ]:



#!pip install keras-rl

#!pip install h5py


# # Import Neccesary library

# In[ ]:


import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


# # Set the relevant variables

# In[ ]:


get_ipython().system('pip install gym[atari]')


# In[ ]:


ENV_NAME = 'Pong-v0'

#get the environment
#extract the number of actions available in the Cartpole problem
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n
print(nb_actions)


# # Building a very simple single hidden layer neural network model.

# In[ ]:


model = Sequential()
model.add(Flatten(input_shape=(1,)  + env.observation_space.shape ))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


# **Next, we configure and compile our agent. We set our policy as Epsilon Greedy and we also set our memory as Sequential Memory because we want to store the result of actions we performed and the rewards we get for each action.**

# In[ ]:


policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)

dqn = DQNAgent(model=model, nb_actions= nb_actions, memory=memory, nb_steps_warmup=100, 
               target_model_update=1e-2, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#trainig...visualization slows down training quite a lot..so putting it off
dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)


# # Test our reinforcement learning model

# In[ ]:


dqn.test(env, nb_episodes=5, visualize=False)


# In[ ]:


get_ipython().system('apt-get install python-opengl -y')
get_ipython().system('pip install pyvirtualdisplay')


# # Playing 

# In[ ]:


from matplotlib import animation , rc
import matplotlib.pyplot as plt

#Run the env
observation = env.reset()

fig = plt.figure()

frame = []
for t in range(50000):
    action = env.action_space.sample()
    state, reward, done,info = env.step(action)
    img = plt.imshow(env.render('rgb_array'))
    frame.append([img])
    if(done):
        break

        
an = animation.ArtistAnimation(fig, frame, interval=100, repeat_delay=1000, blit=True)
rc('animation', html='jshtml')
an


# **More optimiztion is required in order to make the model run more accuratrly**
