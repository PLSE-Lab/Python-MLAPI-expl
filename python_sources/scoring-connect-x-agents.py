#!/usr/bin/env python
# coding: utf-8

# ## Intro
# 
# There were a bunch of questions on how to validate an agent or a neural net for connect 4. So I decided to release a dataset that I use to validate the nets that I use. The dataset was generated with a modified version of the C++ connect4 solver provided by http://connect4.gamesolver.org It contains 1000 samples of board positions from ply 8 to 20. 
# 
# With each of the positions it has the perfect score for the position as well as the scores of all positions after the next move. This allows to estimate how good an agent or a net is by comparing its move with a perfect solution. 
# 
# Format of the dataset: 
# > {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 1, 0, 2, 0, 0, 0, 1, 1, 1, 2, 1, 0, 1, 2], "score": -2, "move score": [-3, -4, -4, -2, -6, -5, -4]}
# 
# Each row is a json dictionary with the following fields:
# * "board": kaggle format of a connect4 baord, 
# * "score": Score for the position, 
# * "move score": Array of 7 scores corresponding to a play in each of the 7 columns
# 
# A note on the scores in the dataset:
# * Score = 0: Game will be a draw
# * Score > 0: Current player will win (the bigger the number the sooner the player will win). The score is the half the ammount of plies from the end the game will be won. So +5 is means the win will be in ply 42 - 2*5 = 32. 
# * Score < 0: Current player will lose (the bigger the number the sooner the player will lose)
# * Score = -99: simply indicates that that was not a legal move.
# 
# There are 2 metrics that I use are:
# * Perfect Move: Here the agent picks a move with the same score as the perfect player.
# * Good Move: The agent picks a move in the same categoty (win, loss or draw) as the perfect player. If an agent play 100% good moves it will play as well as a perfect player, but the win might be later in the game.
# 
# ## Let's analyze the built in agents

# In[ ]:


import json
from kaggle_environments.utils import structify

def win_loss_draw(score):
    if score>0: 
        return 'win'
    if score<0: 
        return 'loss'
    return 'draw'

def score(agent, max_lines = 1000):
    #Scores a connect-x agent with the dataset
    print("scoring ",agent)
    count = 0
    good_move_count = 0
    perfect_move_count = 0
    observation = structify({'mark': None, 'board': None})
    with open("/kaggle/input/1k-connect4-validation-set/refmoves1k_kaggle") as f:
        for line in f:
            count += 1
            data = json.loads(line)
            observation.board = data["board"]
            # find out how many moves are played to set the correct mark.
            ply = len([x for x in data["board"] if x>0])
            if ply&1:
                observation.mark = 2
            else:
                observation.mark = 1
            
            #call the agent
            agent_move = agent(observation,env.configuration)
            
            moves = data["move score"]
            perfect_score = max(moves)
            perfect_moves = [ i for i in range(7) if moves[i]==perfect_score]

            if(agent_move in perfect_moves):
                perfect_move_count += 1

            if win_loss_draw(moves[agent_move]) == win_loss_draw(perfect_score):
                good_move_count += 1

            if count == max_lines:
                break

        print("perfect move percentage: ",perfect_move_count/count)
        print("good moves percentage: ",good_move_count/count)


# In[ ]:


# Score the 2 built in agents
from kaggle_environments import make
env = make("connectx")
# the built in agents are remarkably slow so only evaluating on 100 moves here
score(env.agents["random"],100)  
score(env.agents["negamax"],100)


# Output should be:
# > scoring function random_agent at 0x7f72e753e378  
# > perfect move percentage:  0.22  
# > good moves percentage:  0.67  
# > scoring function negamax_agent at 0x7f72e753e400  
# > perfect move percentage:  0.4  
# > good moves percentage:  0.71  
# 
# 
# Some more references:  
# A neural net that I use in my best agent (1267 score on 2/24/20) score as follows:
# > perfect move percentage:  0.737  
# > good moves percentage:  0.939
# 

# In[ ]:




