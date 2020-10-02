#!/usr/bin/env python
# coding: utf-8

# This Is a pretty basic Agent runs off of three basic principles:
#    1) If there is a clear winning move you should take it
#    2) If there is a clear winning move for your opponent you should block it
#    3) Spaces close to the center tend to be better than those near the edges

# First let's import everything we'll need.

# In[ ]:


from kaggle_environments import make, evaluate
import numpy as np
from random import choice


# The makeBoard function turns the observation list into a 2d array.

# In[ ]:


def makeBoard(obs):
    return np.reshape(obs.board,(6,7))


# The checkWin function (credit to kernel posted by user tanreinama) goes through the grid and checks to see if there are any 4 in a rows vertically, horizontally, or in either diagonal

# In[ ]:


# (credit to https://www.kaggle.com/tanreinama/simple-game-tree-max-depth-4 for checkWin function)
def checkWin(board):
    for y in range(board.shape[0]):
        for x in range(board.shape[1]):
            if x+3 < board.shape[1] and board[y,x] != 0:
                if board[y,x] == board[y,x+1] and                    board[y,x+1] == board[y,x+2] and                    board[y,x+2] == board[y,x+3]:
                    return board[y,x]
            if y+3 < board.shape[0] and board[y,x] != 0:
                if board[y,x] == board[y+1,x] and                    board[y+1,x] == board[y+2,x] and                    board[y+2,x] == board[y+3,x]:
                    return board[y,x]
            if x+3 < board.shape[1] and y+3 < board.shape[0] and board[y,x] != 0:
                if board[y,x] == board[y+1,x+1] and                    board[y+1,x+1] == board[y+2,x+2] and                    board[y+2,x+2] == board[y+3,x+3]:
                    return board[y,x]
            if x+3 < board.shape[1] and y+3 < board.shape[0] and board[y+3,x] != 0:
                if board[y+3,x] == board[y+2,x+1] and                    board[y+2,x+1] == board[y+1,x+2] and                    board[y+1,x+2] == board[y,x+3]:
                    return board[y+3,x]
    return 0


# the ifDropInCol function creates another board simulating what the board would look like if you (or your opponent) picked that column.

# In[ ]:


def ifDropInCol(board, col, player):  
    newBoard=board.copy()
    if newBoard[0,col]==0:
        zeroLoc=0
        for y in range(newBoard.shape[0]):
            if newBoard[y,col]==0:
                zeroLoc=y
        newBoard[zeroLoc,col]=player
        return newBoard


# First for some bookkeeping.  The Agent notes down which player you are and what the opponent is, as well as creates a board from the observation since it's easier to work with.  It then looks and sees which columns still have open space in them and notes them as the possible moves you can mak
# 
# Now the logic begns.  The agent simulates dropping a tile into each space and sees if any would win you the game.  If so, it takes that move.
# 
# If there are no definite winning spaces, the agent checks if dropping an opponent tile into an available space would make them win the game.  If so, you go there instead blocking it.
# 
# If neither of those two methods give a definite best spot to go, it picks a space randomly.  However, This random decision is weighted towards the center being most likely, then the next rows out and so on with the edges being least likely.  This is based on the thought that the center spaces are "Better" since they can be involved in the most 4-in-a-rows.

# In[ ]:


def myAgent(obs, config):    
    # Which player is which?
    myPlayer=obs.mark
    otherPlayer=myPlayer % 2 + 1
    board=makeBoard(obs)
    moves=[c for c in range(config.columns) if obs.board[c]==0]

    ##### Actual Agent Starts Here

    # Find any definite winning Moves
    for m in moves:
        if checkWin(ifDropInCol(board,m,myPlayer))==myPlayer:
            return m
        
    # Find any Defending Moves
    for d in moves:
        if checkWin(ifDropInCol(board,d,otherPlayer))==otherPlayer:
            return d
        
    # Centerload Random Choices so Center is more likely
    if 3 in moves:
        for i in range(8):
            moves.append(3) 
    if 2 in moves:
        for i in range(4):
            moves.append(2)
    if 4 in moves:
        for i in range(4):
            moves.append(4)
    if 1 in moves:
        for i in range(2):
            moves.append(1)
    if 5 in moves:
        for i in range(2):
            moves.append(5)
            
    # Randomly return an available space from the list.
    if len(moves)!=1:
        return choice(moves)
    
    # If theres only 1 space available return it (choice errors out with fewer than 2 options)
    else:
        return moves[0]
    


# Now we wanst to test this against Random and Negamax.  To do that we'll need a reward function.  
# 
# The below is slightly modified from other ones you may see posted , I was having issues with handling of ties and this solved it.

# In[ ]:


def avgReward(x):
    rewards=[a if a[0]!=None else [.5,.5] for a in x]
    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)


# Now to run some simulations, as both first and second player

# In[ ]:


reps=100
randomFirst=evaluate('connectx',[myAgent,'random'], num_episodes=reps)
print("Agent vs Random, Going first: "+str(avgReward(randomFirst)))
randomSecond=evaluate('connectx',['random',myAgent], num_episodes=reps)
print("Agent vs Random, Going second: "+str(1-avgReward(randomSecond)))

#Reduce Reps for Negamax, as it takes a lot longer to run...
reps=20
negamaxFirst=evaluate('connectx',[myAgent,'negamax'], num_episodes=reps)
print("Agent vs Negamax, Going first: "+str(avgReward(negamaxFirst)))
negamaxSecond=evaluate('connectx',['negamax',myAgent], num_episodes=reps)
print("Agent vs Negamax, Going second: "+str(1-avgReward(negamaxSecond)))


# The agent almost always outperforms random, and does better than 50/50 against negamax.  
# The Agent continues to improve as more rules are added, but it also slows down.  For a rules-based approach this is a good starting point.
