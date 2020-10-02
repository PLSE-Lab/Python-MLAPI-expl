#!/usr/bin/env python
# coding: utf-8

# Orginal file and changed mark

# In[ ]:


import gym
env = gym.make('Blackjack-v0')
discount = 1
returns = {}
v = {} # change v to q

def print_dict(a_dict):
    key_list = sorted(a_dict.keys())
    for key in key_list:
        print(key,':',a_dict[key])

#print(env.action_space)
#print(env.observation_space)

#run through 1 or more episodes
for i in range(100):
    #start a new episode
    state = env.reset()
    episode_state = []   #combie state and action 
    episode_action = []  #combie state and action 
    episode_reward = []
    episode_state.append(state) # no need this because we need action
    #print('\nstate',state)
    for t in range(100):
        #get a random action from the environment
        action = env.action_space.sample()
        #print('action',action)
        #send the action to the environment and get back
        #the next state, reward, whether or not the episode is over,
        #and additional information which may or may not be
        #defined in certain environments.
        state, reward, done, info = env.step(action)
        episode_action.append(action) #combie state and action 
        episode_reward.append(reward)
        episode_state.append(state) #combie state and action 
        #print('state',state)
        #print('reward',reward)
        #print('done',done)
        #print('info',info)
        #check for the completion of the episode
        if done:
            #print('Episode finished after {} timesteps'.format(t+1))
            break
    #print(episode_state,'\n',episode_action,'\n',episode_reward)
    g = 0
    for tg in range(len(episode_reward)-1,-1,-1):
        g = discount*g + episode_reward[tg]
        if episode_state.index(episode_state[tg]) == tg:
            if episode_state[tg] in returns:
                returns[episode_state[tg]].append(g)
            else:
                returns[episode_state[tg]] = [g]
            v[episode_state[tg]] = sum(returns[episode_state[tg]])/                                   len(returns[episode_state[tg]])
#print('\nreturns')
#print_dict(returns)
print("\nv")
print_dict(v)
            
#close the environment
env.close()


# In[ ]:


Change v to q


# In[ ]:


import gym
env = gym.make('Blackjack-v0')
discount = 1
returns = {}
q = {} # change v to q

def print_dict(a_dict):
    key_list = sorted(a_dict.keys())
    for key in key_list:
        print(key,':',a_dict[key])

#print(env.action_space)
#print(env.observation_space)

#run through 1 or more episodes
for i in range(10000):
    #start a new episode
    state = env.reset()
    #combie state and action 
    episode_StateAction = []
    episode_reward = []
    #print('\nstate',state)
    for t in range(100):
        #get a random action from the environment
        action = env.action_space.sample()
        #print('action',action)
        #send the action to the environment and get back
        #the next state, reward, whether or not the episode is over,
        #and additional information which may or may not be
        #defined in certain environments.
        episode_StateAction.append((state,action))
        state, reward, done, info = env.step(action)
        episode_reward.append(reward)
        #print('state',state)
        #print('reward',reward)
        #print('done',done)
        #print('info',info)
        #check for the completion of the episode
        if done:
            #print('Episode finished after {} timesteps'.format(t+1))
            break
    #print(episode_state,'\n',episode_action,'\n',episode_reward)
    g = 0
    for tg in range(len(episode_reward)-1,-1,-1):
        g = discount*g + episode_reward[tg]
        if episode_StateAction.index(episode_StateAction[tg]) == tg:
            if episode_StateAction[tg] in returns:
                returns[episode_StateAction[tg]].append(g)
            else:
                returns[episode_StateAction[tg]] = [g]
            q[episode_StateAction[tg]] = sum(returns[episode_StateAction[tg]])/                                   len(returns[episode_StateAction[tg]])
#print('\nreturns')
#print_dict(returns)
print("\nv")
print_dict(q)
            
#close the environment
env.close()

