#!/usr/bin/env python
# coding: utf-8

# # Learning to play Blackjack with Reinforcement Learning
# 
# Over the past couple of years [DeepMind](https://deepmind.com/) have had some amazing results playing [Atari games](https://www.nature.com/articles/nature14236) and [Go](https://www.nature.com/articles/nature16961). While this is fantastic for reinforcement learning, these successes give the impression that it's a new area of machine learning. In fact reinforcement learning has been around for a while, is very well mathematically established and can produce great results on simple games with very little code.
# 
# In this kernel we're going to use [OpenAI Gym](https://gym.openai.com/) and a very basic reinforcement learning technique called Monte Carlo Control to learn how to play Blackjack.
# 
# But first, I want to talk a bit about Gym.

# In[ ]:


import gym

import random
from collections import defaultdict
import numpy as np

# this is all plotting stuff :/
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
get_ipython().run_line_magic('matplotlib', 'inline')

matplotlib.style.use('ggplot')


# ## What is OpenAI Gym?
# 
# Much like how MNIST or ImageNet is used to benchmark different image classifiction techniques, you can think of [OpenAI Gym](https://gym.openai.com/) as a set of environments for reinforcement learning benchmarking.
# 
# There are environments ranging from the very simple (like Blackjack which we are going to use) to the highly complex (robotic hand manipulation, or Atari games). Many of these are "classic" tasks you might see in any reinforcement text book or online course (btw see the end of this kernel for some links) which lets you implement things you find :)
# 
# Here, creating an env is very easy - for example to create the classic [Taxi game](https://gym.openai.com/envs/Taxi-v2/) you just run

# In[ ]:


env = gym.make('Taxi-v2')


# *note: I'm limiting the examples in this kernel to text based envs, there are much more visually interesting kernels out there if you have the ability to set up your kernel the right way*
# 
# To visualise the current state of an env, you simply call `.render()`

# In[ ]:


env.render()


# The objective of this game is to move the yellow block (the taxi) to where a passenger is, pick them up and drop them off in the desired location in as few moves as possible.
# 
# Each env has an action space of things you the user (or your RL agent) can do. In this example there are 6 possible actions

# In[ ]:


env.action_space


# In reinforcement learning we learn what these actions are and how to use these them to carry out the objective.
# 
# When you've met the objective (or the state of the environment is otherwise terminal.. ie you died in pacman) you can restart your env with `.reset()` (this returns the state)

# In[ ]:


env.reset()
env.render()


# To perform an action we can use the `.step()` function with an action (one of the 6 we talked about above)

# In[ ]:


obs, reward, done, _ = env.step(action=0)


# which will then change the state

# In[ ]:


env.render()


# calling `.step()` returns a few things, the new state (here I've called it `obs`), any reward and if the game is complete

# In[ ]:


done


# In[ ]:


reward


# # Whistlestop tour of RL
# 
# There's lots of great resources out there on RL, just a few are linked at the end of this kernel, so I don't want to dwell on this too long, but I'd feel wrong publishing this kernel without a few sentences about what RL is all about.
# 
# Think of a task, like driving a car or playing an Atari game - now how would you decide if you did well at that task? In RL we do this by measuring a "reward", in fact one of the central ideas of RL is the success in any task can be measured with a single scalar number. Our job in RL is simply to maximise this number.
# 
# Defining this reward is easy for games (points in an Atari game, or +1 points for a win/-1 for a loss, etc) which is part the reason they are studied so much.
# 
# We do this by visualising any task like so
# 
# ![rl](https://lilianweng.github.io/lil-log/assets/images/RL_illustration.png)
# 
# *- image borrowed from [this awesome blog post](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)*
# 
# Here we train some "agent" to look at the environment (like Gym) and based upon the state maximise some reward.
# 
# How the agent decides what action to take in what state is called a policy.
# 
# There are a lot of ways to do this, here I'm going to implement one of the most simple approaches.
# 
# 
# ## Monte-Carlo Control
# 
# You can think of Monte Carlo Control as a method of approximating optimal policies and it works as follows:
# 
# - we keep a function which gives us the approximate "value" of being in a given state (typically called the Q-function), we'll represent this with a dict and initialise it randomly. You can think of this as "if I find myself in state s, what reward can I expect at the end of the game? how good or bad is it to be there?"
# - we'll then start playing our game (blackjack). When we start the game or take some action Gym will give us a state (our hand, the dealers open hand, if the game is complete etc).
# - using the value function, we decide our next action by looing at the available actions and picking the one with the highest value (we will inject some randomness here, which I'll talk about shortly)
# - at the end of the game we update the value function with what we learnt
# 
# This sounds basic, but as you play more and more games and you explore more and more states you will eventually reach the best possible policy
# 
# There are a couple of nice features of learning like this:
# 
# - it's model free, meaning we're not going to explicitly build an internal model of how Blackjack works and use that to help, instead we are going to learn by playing
# - it's on policy, meaning we're going to "learn on the job", learn about a policy by experience with that policy

# In[ ]:


# start a new blackjack env
env = gym.make('Blackjack-v0')

# number of games to play
episodes = 500000

# sometimes you may want to discount rewards, I'm not going to cover this here
gamma = 1.


# In[ ]:


def get_epsilon(N_state_count, N_zero=100):
    """
    This is our function to calculate epsilon and is core to how we are going to pick our next action.
    
    When we first start exploring our state-action space, we have little or no knowledge of the environment, meaning we
    have little or no knowledge about what a good action might be. In this case we want to pick a random action (ie we
    want a large epsilon). As our knowledge gets better, we can have more confidence in what we're doing and so we'd like
    to pick what we know is a good action more often.
    
    We're initialising N_zero to 100, but this is a hyperparameter we can tune
    """
    return N_zero / (N_zero + N_state_count)


# In[ ]:


def get_action(Q, state, state_count, action_size):
    """
    Given our value function (Q) and state, what action should we take?
    
    If we haven't seen this state before we should pick an action at random, after all we have no information about
    what action might be best.
    
    If we have infinite experience, what should we do? In this case we would like to pick the action with the
    highest expected value all the time.
    
    To fulfil this we're going to use what is known as GLIE (Greedy in the Limit of Infinite Exploration).
    
    The idea is we pick the action with the highest expected value with a probability of `1 - epsilon` and a random
    action with probability epsilon. At first epsilon is large but it eventually decays to zero as we play an infinite
    number of games.
    
    Doing this guarentees we visit all possible states and actions.
    """
    random_action = random.randint(0, action_size - 1)
    
    best_action = np.argmax(Q[state])
        
    epsilon = get_epsilon(state_count)
    
    return np.random.choice([best_action, random_action], p=[1. - epsilon, epsilon])


# In[ ]:


def evaluate_policy(Q, episodes=10000):
    """
    Helper function which helps us evaluate how good our policy is.
    
    We do this by playing 10000 games of blackjack and returning the win ratio.
    """
    wins = 0
    for _ in range(episodes):
        state = env.reset()
        
        done = False
        while not done:
            action = np.argmax(Q[state])
            
            state, reward, done, _ = env.step(action=action)
            
        if reward > 0:
            wins += 1
        
    return wins / episodes


# In[ ]:


def monte_carlo(gamma=1., episodes=5000, evaluate=False):

    # this is our value function, we will use it to keep track of the "value" of being in a given state
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # to decide what action to take and calculate epsilon we need to keep track of how many times we've
    # been in a given state and how often we've taken a given action when in that state
    state_count = defaultdict(float)
    state_action_count = defaultdict(float)

    # for keeping track of our policy evaluations (we'll plot this later)
    evaluations = []

    for i in range(episodes):
        # evaluating a policy is slow going, so let's only do this every 1000 games
        if evaluate and i % 1000 == 0:
            evaluations.append(evaluate_policy(Q))
    
        # to update our value function we need to keep track of what states we were in and what actions
        # we took throughout the game
        episode = []
    
        # lets start a game!
        state = env.reset()
        done = False
    
        # and keep playing until it's done (recall this is something Gym will tell us)
        while not done:
            # so we're in some state, let's remember we've been here and pick an action using our
            # function defined above
            state_count[state] += 1
            action = get_action(Q, state, state_count[state], env.action_space.n)

            # when we take that action, recall Gym will give us a new state, some reward and if we are done
            new_state, reward, done, _ = env.step(action=action)
        
            # save what happened, we're just going to keep the state, action and reward
            episode.append((state, action, reward))
        
            state = new_state

        # at this point the game is finished, we either won or lost
        # so we need to take what happened and update our value function
        G = 0
    
        # because you can only win or lose a game of blackjack we only get a reward at the end of the game
        # (+1 for a win, 0 for a draw, -1 for a loss). So let's start at the end of the game and work
        # backwards through our states to decide how good it was to be in a state
        for s, a, r in reversed(episode):
            new_s_a_count = state_action_count[(s, a)] + 1
            
            # we need some way of deciding how the game we just played impacted our value function. The
            # standard approach here is to take the reward(s) we got playing over multiple games and
            # taking the mean. We can update the mean as we go using what is known as incremental averaging
            # https://math.stackexchange.com/questions/106700/incremental-averageing
            G = r + gamma * G
            state_action_count[(s, a)] = new_s_a_count
            Q[s][a] = Q[s][a] + (G - Q[s][a]) / new_s_a_count
            
    return Q, evaluations


# And that's it!
# 
# So let's run it for half a million games!

# In[ ]:


Q_mc, evaluations = monte_carlo(episodes=500000, evaluate=True)


# Great!
# 
# But what use is this? Well we can use our value function to plot how good it is to be in a given state (with a greedy policy)
# 
# Adapted from this [function](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/plotting.py)

# In[ ]:


def plot_value_function(Q, title="Value Function"):
    V = defaultdict(float)

    for state, action_rewards in Q.items():
        r1, r2 = action_rewards
        action_value = np.max([r1, r2])
        V[state] = action_value
    
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player sum')
        ax.set_ylabel('Dealer showing')
        ax.set_zlabel('Value')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title)
        ax.view_init(ax.elev, 120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "value function")
    plot_surface(X, Y, Z_ace, "value function - usable ace")


# In[ ]:


plot_value_function(Q_mc)


# We have two plots here, one for when you have an ace (an ace can be 1 or 10) and one for when you don't - notice there's an increase in the win rate around 10/11 when you don't have an ace - that's because the next card you get might be an ace and take you straight to 21!
# 
# Recall we also evaluated our policy as we went along, lets plot how good it got over time

# In[ ]:


plt.plot([i * 1000 for i in range(len(evaluations))], evaluations)
plt.xlabel('episode')
plt.ylabel('win rate')


# Notice this is surprisingly noisey! Sadly that's a general feature of reinforcement learning

# ## Next steps and acknowledgements
# 
# This kernel is kind of a work in progress, I'd eventually like to explore the optimal policy more and use what is known as TD learning to achive the same result (hopefully in fewer games!)
# 
# If you're interested in learning more about RL then there are lots of fantastic resources out there, including:
# 
# - [Reinforcement Learning: An Introduction (2nd edition)](http://incompleteideas.net/book/the-book-2nd.html) by Sutton and Barto
# - [David Silver's Reinforcement Learning course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
# - [DeepMind's UCL lecture course](https://www.youtube.com/watch?v=iOh7QUZGyiU&list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)
# - [OpenAI's spinning up in Deep RL resource](https://blog.openai.com/spinning-up-in-deep-rl/)
# 
# This kernel is based upon a homework which was part of Prof Silver's course and a problem in Sutton and Barto's book.
# 
# The code for plotting the value function was a bit beyond my matplotlib skills, luckily I stumbled across [this github project](https://github.com/dennybritz/reinforcement-learning) which implemented nearly exactly what was needed. So thank you there.
# 
# Also, I want to point people to [this great blogpost](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#sarsa-on-policy-td-control) from which I borrowed a few images - thank you!

# In[ ]:




