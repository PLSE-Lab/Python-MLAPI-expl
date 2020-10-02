#!/usr/bin/env python
# coding: utf-8

# In this notebook I'd like to explore how the environment is created and how it is passed to the agent.
# 
# ![](https://i.imgur.com/loGXjIN.png)
# 
# After inspecting the environment, I give some pointers how to build a good agent with the current version of the kaggle Docker and some sources when Pytorch is available to play around with.
# 
# The environment itself give you a lot of info to work with, and we'll build a 'slightly better than random' agent.
# 
# This kernel is based on the [Getting Started](https://www.kaggle.com/ajeffries/connectx-getting-started) notebook.

# In[ ]:


# 1. Enable Internet in the Kernel (Settings side pane)

# 2. Curl cache may need purged if v0.1.4 cannot be found (uncomment if needed). 
# !curl -X PURGE https://pypi.org/simple/kaggle-environments

# ConnectX environment was defined in v0.1.4
get_ipython().system("pip install 'kaggle-environments>=0.1.4'")


# # Investigate ConnectX Environment
# Let's investigate, what Kaggle will be doing with this environment:

# In[ ]:


from kaggle_environments import evaluate, make

env = make("connectx", debug=True)


# There are some agents coming baked in, namely the `random` agent, that will create a baseline, if your final agent is doing better than flipping a coin. The `negamax` agent? We'll talk about that one below!

# In[ ]:


env.agents


# In[ ]:


env.configuration


# So clearly we can vary columns and rows, as well as the amount of tokens in a line to win. I guess this may be a nice test case for bigger games.
# 
# But there's also the amount of steps and a timeout variable. I'll venture a guess and say that your move is a maximum of 2 seconds.
# 
# With the specification commend, you'll be able to get a condensed dictionary that has most of the important information!

# In[ ]:


env.specification


# We get the configuration from before and a bit of fluff. However, there's also the reward with the following values:
# 
# - Loss: 0
# - Draw: 0.5
# - Win: 1
# 
# and a good description of what to do as a valid action. Choose a column to drop your token in.

# # Inspect inner Agent workings
# 
# Let's create a little agent to inspect what objects we're even working with in this gym environment.

# In[ ]:


# This agent random chooses a non-empty column.
def my_agent(observation, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])


# And inspect the first iteration:

# In[ ]:


# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()

print("Observation contains:\t", observation)
print("Configuration contains:\t", env.configuration)

my_action = my_agent(observation, env.configuration)
print("My Action", my_action)
observation, reward, done, info = trainer.step(my_action)
# env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render(mode="ipython", width=100, height=90, header=False, controls=False)
print("Observation after:\t", observation)
#env.render()


# From this it is clear that the board contains a flattened 1D array of the $7 \times 6$ board as described in the specification. Zero contains non-occupied spaces, our tokens represent a one and our opponent has value token 2.

# In[ ]:


def my_comatose_agent(observation, configuration):
    from random import choice
    from time import sleep
    sleep(2)
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

def my_sleepy_agent(observation, configuration):
    from random import choice
    from time import sleep
    sleep(1)
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])


# In[ ]:


print(evaluate("connectx", [my_comatose_agent, "random"], num_episodes=1))
print(evaluate("connectx", [my_sleepy_agent, "random"], num_episodes=1))
print(evaluate("connectx", [my_agent, "random"], num_episodes=1))


# So each move has to be sub two seconds to be valid. This has some serious implications considering that we're in Python and most of these game strategies rely on some sort of minimax game. That means, we have to somewhat traverse the game-state space to know how well we're doing and that is costly. So essentially, play all hypothetical games and then make the optimal choice on that calculation.

# # Taking The Hint
# 
# If you look in the starter notebook, you'll see that the evaluation is done against the following code:

# In[ ]:


def mean_reward(rewards):
    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)

print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=3)))


# The important bit is the keyword `negamax`. It's a [special version of the minimax](https://en.wikipedia.org/wiki/Negamax) strategy, that optimizes based on the symmetry, that in this two player game you are always doing better when your opponent is doing worse. So essentially, the game state is always "I'm at score `0.7` so my opponent is at `0.3` -- then you're at `0.4` and they're at `0.6`. Just a fact from a two player game with [perfect information](https://en.wikipedia.org/wiki/Perfect_information).
# 
# ![](https://i.imgur.com/TBZXsYA.gif)
# CC-BY-SA 3.0 [Maschelos](https://en.wikipedia.org/wiki/File:Plain_Negamax.gif)
# 
# If you're interested in implementing your own Negamax strategy with all the optimizations, I find [this tutorial](http://blog.gamesolver.org/solving-connect-four/03-minmax/) exceptional, despite being in C++. These optimizations probably include [Alpha-Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) and [Transposition Tables](https://en.wikipedia.org/wiki/Transposition_table).
# 
# Personally, I found the [EasyAI](https://github.com/Zulko/easyAI) implementation pretty understandable and worth diving into. 
# 
# Let's have a look at how kaggle approaches the negamax problem:

# In[ ]:


import inspect
import os

print(inspect.getsource(env.agents['negamax']))


# How well is negamax doing against itself?

# In[ ]:


neg_v_neg = evaluate("connectx", [env.agents['negamax'], "negamax"], num_episodes=10)
print(neg_v_neg)
print(mean_reward(neg_v_neg))


# That's odd...

# # Reinforcement Learning
# As of writing this kernel, the challenge is limited to Standard Python, gym, numpy and scipy. So all the nice things in pytorch are not available yet. Once that is available, you can go on to some cool shenanigans essentially [mimicing AlphaZero](https://towardsdatascience.com/from-scratch-implementation-of-alphazero-for-connect4-f73d4554002a)
# 
# ![](https://miro.medium.com/max/850/1*4jBLXRsNVeOMBhOqO-8v8w.png)
# 
# Until then, we'll have to do something different. Let's try and do a bit better than random.

# # Let's Try Something Stupid
# How about, we try random choice, but just not take the step that will make us lose?
# 
# And yes, this could be the first step toward implementing negamax. Considering, you have to simulate the games in a copy of the environment.

# In[ ]:


def try_not_to_loose_agent(observation, configuration):
    from random import choice
    from kaggle_environments import make
    env = make("connectx", debug=True)
    trainer = env.train([None, "negamax"])
    
    cols = list(range(configuration.columns))
    while cols:
        # We set the state of the environment, so we can experiment on it.
        env.state[0]['observation'] = observation
        env.state[1]['observation'] = observation
        # Take a random column that is not full
        my_action = choice([c for c in cols if observation.board[c] == 0])
        # Simulate the next step
        out = env.train([None, "negamax"]).step(my_action)
        # If the next step makes us lose, take a different step!
        if out[2]:
            cols.pop(my_action)
        else:
            return my_action
    else:
        # If we run out of steps to take, we just loose with one step.
        return 1


# In[ ]:


stupid_v_random = evaluate("connectx", [try_not_to_loose_agent, "random"], num_episodes=10)
print(stupid_v_random)
print(mean_reward(stupid_v_random))

stupid_v_neg = evaluate("connectx", [try_not_to_loose_agent, "negamax"], num_episodes=10)
print(stupid_v_neg)
print(mean_reward(stupid_v_neg))


# In[ ]:


import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(try_not_to_loose_agent, "submission.py")


# ## Good luck!
# Hope you find something interesting to work with!

# In[ ]:




