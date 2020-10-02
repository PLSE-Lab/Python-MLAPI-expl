#!/usr/bin/env python
# coding: utf-8

# # Crossentropy method
# This notebook will teach you to solve reinforcement learning problems with crossentropy method. We'll follow-up by scaling everything up and using . neural network policy.

# In[ ]:


# In Google Colab, uncomment this:
get_ipython().system('wget https://bit.ly/2FMJP5K -O setup.py && bash setup.py')


# In[ ]:


# XVFB will be launched if you run on a server
import os
if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY")) == 0:
    get_ipython().system('bash ../xvfb start')
    os.environ['DISPLAY'] = ':1'


# In[ ]:


import gym
import numpy as np
import pandas as pd

env = gym.make("Taxi-v3")
env.reset()
env.render()


# In[ ]:


n_states = env.observation_space.n
n_actions = env.action_space.n

print("n_states=%i, n_actions=%i" % (n_states, n_actions))


# # Create stochastic policy
# This time our policy should be a probability distribution
# $policy[s, a]=P(\text{take action a | in state s})$
# 
# Since we still use int3eger state and action representations, you can use a 2-dimensional array to represent the policy.
# 
# Please initialize policy uniformly, that is, probabilities of all actions sh9oould be equal.

# In[ ]:


policy = np.ones((n_states, n_actions))/(n_actions)


# In[ ]:


assert type(policy) in (np.ndarray, np.matrix)
assert np.allclose(policy, 1./n_actions)
assert np.allclose(np.sum(policy, axis=1), 1)


# # Play the game
# Just like before, but we also record all states and actions we took.

# In[ ]:


def generate_session(policy, t_max=10**4):
    """
    Play game until end or for t_max ticks.
    :param policy: an array of shape [n_states,n_actions] with action probabilities
    :returns: list of states, list of actions and sum of rewards
    """
    states, actions = [], []
    total_reward = 0.

    s = env.reset()

    for t in range(t_max):

        #a = <sample action from policy(hint: use np.random.choice) >
        a = np.random.choice(n_actions, 1, p=policy[s])[0]

        new_s, r, done, info = env.step(a)

        # Record state, action and add up reward to states,actions and total_reward accordingly.
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break
    return states, actions, total_reward


# In[ ]:


s, a, r = generate_session(policy)
assert type(s) == type(a) == list
assert len(s) == len(a)
assert type(r) in [float, np.float]


# In[ ]:


# let's see the initial reward distribution
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sample_rewards = [generate_session(policy, t_max=1000)[-1] for _ in range(200)]

plt.hist(sample_rewards, bins=20)
plt.vlines([np.percentile(sample_rewards, 50)], [0], [100], label="50'th percentile", color='green')
plt.vlines([np.percentile(sample_rewards, 90)], [0], [100], label="90'th percentile", color='red')
plt.legend()


# # Crossentropy method steps

# In[ ]:


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    Please return elite states and actions in their original order 
    [i.e. sorted by session number and timestep within session]

    If you are confused, see examples below. Please don't assume that states are integers
    (they will become different later).
    """

#     reward_threshold = <Compute minimum reward for elite sessions. Hint: use np.percentile >
    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_states = []
    elite_actions = []
    for i in range(len(rewards_batch)):
        if rewards_batch[i] >= reward_threshold:
            elite_states = elite_states + states_batch[i]
            elite_actions = elite_actions +actions_batch[i]
    return elite_states, elite_actions


# In[ ]:


states_batch = [
    [1, 2, 3],     # game1
    [4, 2, 0, 2],  # game2
    [3, 1],        # game3
]

actions_batch = [
    [0, 2, 4],     # game1
    [3, 2, 0, 1],  # game2
    [3, 3],        # game3
]
rewards_batch = [
    3,  # game1
    4,  # game2
    5,  # game3
]

test_result_0 = select_elites(
    states_batch, actions_batch, rewards_batch, percentile=0)
test_result_40 = select_elites(
    states_batch, actions_batch, rewards_batch, percentile=30)
test_result_90 = select_elites(
    states_batch, actions_batch, rewards_batch, percentile=90)
test_result_100 = select_elites(
    states_batch, actions_batch, rewards_batch, percentile=100)

assert np.all(test_result_0[0] == [1, 2, 3, 4, 2, 0, 2, 3, 1])      and np.all(test_result_0[1] == [0, 2, 4, 3, 2, 0, 1, 3, 3]),    "For percentile 0 you should return all states and actions in chronological order"
assert np.all(test_result_40[0] == [4, 2, 0, 2, 3, 1]) and     np.all(test_result_40[1] == [3, 2, 0, 1, 3, 3]),    "For percentile 30 you should only select states/actions from two first"
assert np.all(test_result_90[0] == [3, 1]) and     np.all(test_result_90[1] == [3, 3]),    "For percentile 90 you should only select states/actions from one game"
assert np.all(test_result_100[0] == [3, 1]) and    np.all(test_result_100[1] == [3, 3]),    "Please make sure you use >=, not >. Also double-check how you compute percentile."
print("Ok!")


# In[ ]:


def update_policy(elite_states, elite_actions):
    """
    Given old policy and a list of elite states/actions from select_elites,
    return new updated policy where each action probability is proportional to

    policy[s_i,a_i] ~ #[occurences of si and ai in elite states/actions]

    Don't forget to normalize policy to get valid probabilities and handle 0/0 case.
    In case you never visited a state, set probabilities for all actions to 1./n_actions

    :param elite_states: 1D list of states from elite sessions
    :param elite_actions: 1D list of actions from elite sessions

    """

    new_policy = np.zeros([n_states, n_actions])

#     <Your code here: update probabilities for actions given elite states & actions >
    # Don't forget to set 1/n_actions for all actions in unvisited states.
    for e_i in range(len(elite_states)):
        # Add 1 for every elite state and action
        new_policy[elite_states[e_i], elite_actions[e_i]] += 1
    # check for each state
    for s_i in range(n_states):
        total_actions = sum(new_policy[s_i])
        if total_actions != 0:
            new_policy[s_i] = new_policy[s_i]/total_actions
        else:
            new_policy[s_i] = np.ones(n_actions)/n_actions
    return new_policy


# In[ ]:


elite_states = [1, 2, 3, 4, 2, 0, 2, 3, 1]
elite_actions = [0, 2, 4, 3, 2, 0, 1, 3, 3]

new_policy = update_policy(elite_states, elite_actions)

assert np.isfinite(new_policy).all(
), "Your new policy contains NaNs or +-inf. Make sure you don't divide by zero."
assert np.all(
    new_policy >= 0), "Your new policy can't have negative action probabilities"
assert np.allclose(new_policy.sum(
    axis=-1), 1), "Your new policy should be a valid probability distribution over actions"
reference_answer = np.array([
    [1.,  0.,  0.,  0.,  0.],
    [0.5,  0.,  0.,  0.5,  0.],
    [0.,  0.33333333,  0.66666667,  0.,  0.],
    [0.,  0.,  0.,  0.5,  0.5]])
assert np.allclose(new_policy[:4, :5], reference_answer)
print("Ok!")


# # Training loop
# Generate sessions, select N best and fit to those.

# In[ ]:


from IPython.display import clear_output

def show_progress(rewards_batch, log, percentile, reward_range=[-990, +10]):
    """
    A convenience function that displays training progress. 
    No cool math here, just charts.
    """

    mean_reward = np.mean(rewards_batch)
    threshold = np.percentile(rewards_batch, percentile)
    log.append([mean_reward, threshold])

    clear_output(True)
    print("mean reward = %.3f, threshold=%.3f" % (mean_reward, threshold))
    plt.figure(figsize=[8, 4])
    plt.subplot(1, 2, 1)
    plt.plot(list(zip(*log))[0], label='Mean rewards')
    plt.plot(list(zip(*log))[1], label='Reward thresholds')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.hist(rewards_batch, range=reward_range)
    plt.vlines([np.percentile(rewards_batch, percentile)],
               [0], [100], label="percentile", color='red')
    plt.legend()
    plt.grid()

    plt.show()


# In[ ]:


# reset policy just in case
policy = np.ones([n_states, n_actions]) / n_actions


# In[ ]:


n_sessions = 250  # sample this many sessions
percentile = 50  # take this percent of session with highest rewards
learning_rate = 0.5  # add this thing to all counts for stability

log = []

for i in range(100):

#     %time sessions = [ < generate a list of n_sessions new sessions > ]
    get_ipython().run_line_magic('time', 'sessions = [generate_session(policy) for _ in range(n_sessions)]')
    
    states_batch, actions_batch, rewards_batch = zip(*sessions)

#     elite_states, elite_actions = <select elite states/actions >
    elite_states, elite_actions = select_elites(states_batch=states_batch, actions_batch=actions_batch, rewards_batch=rewards_batch, percentile=percentile)

#     new_policy = <compute new policy >
    new_policy = update_policy(elite_states=elite_states, elite_actions=elite_actions)
    
    policy = learning_rate*new_policy + (1-learning_rate)*policy

    # display results on chart
    show_progress(rewards_batch, log, percentile)


# In[ ]:




