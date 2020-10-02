#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Q-Learning from Scratch in Python with OpenAI Gym
# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

# In[ ]:



import gym
import time
from IPython.display import clear_output

env = gym.make("Taxi-v3").env

# for i in range(0,100):
#     clear_output(wait=True)
#     env.reset()
#     env.render()
#     time.sleep(0.5)
    


# In[ ]:


env.s = 328
env.render()
print(env.step(2))
time.sleep(10)
clear_output(wait=True)
env.render()


# In[ ]:


env.P


# # Solution without RL Algorithm
# Taking random actions from each state

# In[ ]:


epochs = 0
penalties, reward = 0, 0

frames = [] # for animation

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1
    
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'episode': '0',
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1
    
    
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


# ## Printing frames

# In[ ]:


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Episode: {frame['episode']}")
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        time.sleep(1)


# In[ ]:


print_frames(frames)


# # Reinforcement Learning using Q-Learning

# In[ ]:


import numpy as np
q_table = np.zeros([env.observation_space.n, env.action_space.n])


# In[ ]:


get_ipython().run_cell_magic('time', '', '"""Training the agent"""\n\nimport random\nfrom IPython.display import clear_output\n\n# Hyperparameters\nalpha = 0.1\ngamma = 0.6\nepsilon = 0.1\n\n# For plotting metrics\nall_epochs = []\nall_penalties = []\n\nfor i in range(1, 100001):\n    state = env.reset()\n\n    epochs, penalties, reward, = 0, 0, 0\n    done = False\n    \n    while not done:\n        if random.uniform(0, 1) < epsilon:\n            action = env.action_space.sample() # Explore action space\n        else:\n            action = np.argmax(q_table[state]) # Exploit learned values\n\n        next_state, reward, done, info = env.step(action) \n        \n        old_value = q_table[state, action]\n        next_max = np.max(q_table[next_state])\n        \n        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)\n        q_table[state, action] = new_value\n\n        if reward == -10:\n            penalties += 1\n\n        state = next_state\n        epochs += 1\n        \n    if i % 100 == 0:\n        clear_output(wait=True)\n        print(f"Episode: {i}")\n\nprint("Training finished.\\n")')


# In[ ]:


q_table[328]


# # Evaluate agent's performance after Q-learning

# In[ ]:


total_epochs, total_penalties = 0, 0
episodes = 100
frames = []

for ep in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1
        
        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'episode': ep, 
            'state': state,
            'action': action,
            'reward': reward
            }
        )
        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")


# # Visualization

# In[ ]:


print_frames(frames)


# In[ ]:




