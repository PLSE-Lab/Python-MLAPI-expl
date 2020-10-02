#!/usr/bin/env python
# coding: utf-8

# # A More Useful Negamax Opponent
# ## Introduction
# 
# Let's say you've built multiple agents for the ConnectX competition.  Is there a relatively simple way to determine which one is best?
# 
# You could always submit them all to the competition and see which one climbs the leaderboard.  Or you could write your own tournament software and let them play one another.
# 
# But to me the most straight-forward way would be to have them all play against a common opponent and see which one is the best.

# ## Grading Agents
# 
# I'm going to give each agent a grade when playing against a given opponent.  The grade_agent function below is similar to others that have been posted.  Rather than returning the number of wins, loses and draws, I'm going to return the percentage of rewards won.  (This is what we want to know in zero-sum games like ConnectX).  This will evaluate the agent by playing an equal number of games as both player 1 and player 2.

# In[ ]:


get_ipython().system("pip install 'kaggle-environments'")


# In[ ]:


from kaggle_environments import evaluate
from operator import itemgetter

def grade_agent(game_name, agent, opponent, episodes):
    as_p1 = evaluate(game_name, [agent, opponent], num_episodes=episodes)
    as_p1_reward = sum(map(itemgetter(0), as_p1))
    as_p1_total = sum(map(itemgetter(1), as_p1)) + as_p1_reward
    
    as_p2 = evaluate(game_name, [opponent, agent], num_episodes=episodes)
    as_p2_reward = sum(map(itemgetter(1), as_p2))
    as_p2_total = sum(map(itemgetter(0), as_p2)) + as_p2_reward
    
    return 100 * (as_p1_reward + as_p2_reward) / (as_p1_total + as_p2_total)


# ### Versus Random
# 
# I built two agents - agent_A and agent_B.  The screenshot below shows how they did versus the common opponent random.
# 

# ```python
# a_vs_random = grade_agent("connectx", agent_A, "random", 50)
# b_vs_random = grade_agent("connectx", agent_B, "random", 50)
# 
# print("Agent A v Random", a_vs_random)
# print("Agent B v Random", b_vs_random)
# ```
# 
#     Agent A v Random 100.0
#     Agent B v Random 100.0
# 
# 

# #### Issues
# 
# They both scored 100.  That's good.  But it is not at all helpful in determining which agent is better.  The problem is simply that the random agent is not good.

# ### Versus Negamax
# 
# The negamax agent is better than random.  Let's see what happens when my agents play against negamax.

# ```python
# a_vs_negamax = grade_agent("connectx", agent_A, "negamax", 50)
# b_vs_negamax = grade_agent("connectx", agent_B, "negamax", 50)
# 
# print("Agent A v Negamax", a_vs_negamax)
# print("Agent B v Negamax", b_vs_negamax)
# ```
# 
#     Agent A v Negamax 100.0
#     Agent B v Negamax 100.0
# 
# 

# #### Issues
# 
# This looks depressingly similar.  We have not recieved any information that let's us distingish between the play of agent_A and agent_B.
# 
# But the situation is even worse than that.  Negamax is a deterministic agent.  By that I mean that negamax will always play the same moves every game against an opponent unless that opponent plays different moves.  My agents A and B are both deterministic too.  This is a problem.
# 
# For agent_A we played 100 games vs negamax (50 as player 1 and 50 as player 2).  But playing multiple games did not give us any additional information, because **all of the games looked the same**.  My agent was always playing the same moves and negamax was always playing the same moves.  We could have gotten the same information by playing 2 games instead (one as player 1 and one as player 2).  This paragraph is obviously also true for agent_B.
# 
# So really getting a score from grade_agent() with a deterministic agent versus negamax (or any other deterministic agent) can only give you 5 values.
# 
# - 100.0 - you win as both player 1 and player 2
# - 75.0 - you win as player 1 and tie as player 2 (or vice-versa)
# - 50.0 - you win as player 1, but lose as player 2 (or vice-versa), or you tie twice
# - 0.25 - you lose as player 1, but tie as player 2 (or vice-versa)
# - 0.0 - you lose as both player 1 and player 2
# 
# It looks like playing against negamax can't tell you much either.
# 

# ## Combining Agents
# 
# It would be nice if negamax didn't play the same game every time.  If it varied we could play negamax multiple times and get different results.  But that is not what negamax is written to do.
# 
# One solution would be if we had a combined agent that played like negamax most of the time, but occasionally played random moves so that our agent would encounter different board positions.

# In[ ]:


import random

def combined_agent(default_agent, alternate_agent, epsilon):
    def updated_agent(obs, config):
        if (random.random() < epsilon):
            return alternate_agent(obs, config)
        return default_agent(obs,config)
    return updated_agent


# ### $\epsilon$-Greedy Negamax
# 
# Let's make a new e_greedy_negamax that plays like negamax most of the time, but occasionally makes random moves.  The e_greedy_negamax will **not** be better than negamax, because it will occasionally move randomly.  It will be a **more useful** opponent, because it will play relatively well and it will occasionally present my agents with different board positions.
# 

# In[ ]:


from kaggle_environments.envs.connectx.connectx import negamax_agent
from kaggle_environments.envs.connectx.connectx import random_agent

e_greedy_negamax = combined_agent(negamax_agent, random_agent, 0.2)


# I don't know if 0.2 sounds too high for you or not.  The longest possible game is 21 moves by each player.  An epsilon of 0.2 means that e_greedy_negamax will play a full game with no random moves about 1% of the time.  If the game ends after only 10 moves by each player than e_greedy_negamax will play with no random moves about 11% of the time.  It sounds reasonable to me.
# 
# Let's try this out.

# ```python
# a_vs_eg_negamax = grade_agent("connectx", agent_A, e_greedy_negamax, 50)
# b_vs_eg_negamax = grade_agent("connectx", agent_B, e_greedy_negamax, 50)
# 
# print("Agent A v e-greedy Negamax", a_vs_eg_negamax)
# print("Agent B v e-greedy Negamax", b_vs_eg_negamax)
# ```
# 
#     Agent A v e-greedy Negamax 76.0
#     Agent B v e-greedy Negamax 90.0
# 
# 

# Playing an $\epsilon$-greedy negamax has given us some useful information.  Agent B appears to be much better than agent A.
