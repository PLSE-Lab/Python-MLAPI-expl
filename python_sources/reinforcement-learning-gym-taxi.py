#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym


# * blue = passenger
# * purple = destination
# * yellow = empty taxi
# * green = full taxi
# * RGBY = location for destination and passanger

# In[ ]:


env = gym.make("Taxi-v2").env
env.render()


# * reset env and return  random initial state

# In[ ]:


env.reset()


# In[ ]:


print(env.observation_space)
print(env.action_space)


# * taxi row, taxi column, passenger index, destination

# In[ ]:


state = env.encode(3,1,2,3)
print(state)


# In[ ]:


env.s = state
env.render()


# There are 6 discrete deterministic actions:
#     - 0: move south
#     - 1: move north
#     - 2: move east 
#     - 3: move west 
#     - 4: pickup passenger
#     - 5: dropoff passenger
#     
# First column:probability, 
# Second column:next_state, 
# Third column:reward, 
# Fourt column:done

# In[ ]:


env.P[331]


# Again, we must reset our environment.

# In[ ]:


env.reset()


# In[ ]:


total_reward_list = []
# episode
for j in range(5):
    env.reset()
    time_step = 0
    total_reward = 0
    list_visualize = []
    while True:
        time_step += 1
        #choose action
        action = env.action_space.sample()
        #perform action and get reward
        state, reward, done, _ =  env.step(action) # state = next state
        #total reward
        total_reward += reward
        # visualize
        list_visualize.append({"frame": env.render(mode = "ansi"),
                                "state": state, "action": action, "reward":reward,
                                "Total Reward": total_reward})
        if done:
            total_reward_list.append(total_reward)
            break


# In[ ]:


import time       
for i, frame in enumerate(list_visualize):
    print(frame["frame"])
    print("Timestep: ", i + 1)
    print("State: ", frame["state"])
    print("action: ", frame["action"])
    print("reward: ", frame["reward"])
    print("Total Reward: ", frame["Total Reward"])
    # time.sleep(2)

