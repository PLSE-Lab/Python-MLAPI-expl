#!/usr/bin/env python
# coding: utf-8

# * Some heuristic reward/penalty system is developed
# * One step ahaed calculation strategy is implemented (sim_play).
# * Deeper calculation is implemented (play_deep), but previous one (sim_play) got better reulsts.
# * Developed agent (sim_play) has 99.9% winning rate against random.
# 
# forked from https://www.kaggle.com/domcastro/connectx-rule-based-easy-solution-or-negamax
# 

# In[ ]:


# 1. Enable Internet in the Kernel (Settings side pane)

# 2. Curl cache may need purged if v0.1.4 cannot be found (uncomment if needed). 
# !curl -X PURGE https://pypi.org/simple/kaggle-environments

# ConnectX environment was defined in v0.1.4
get_ipython().system("pip install 'kaggle-environments>=0.1.4'")


# In[ ]:


from kaggle_environments import evaluate, make
from kaggle_environments.envs.connectx import connectx as ctx

env = make("connectx", debug=True)
env.render()


# # Create an Agent

# In[ ]:


def agent(observation, configuration):
    import numpy as np
    
    # reward parameters
    rw_win = 10   # reward for win
    rw_loss = -15 # penalty for loose
    rw_3=2        # reward for making 3 in a row
    rw_3enemy=-3  # penalty for making 3 in a row (enemy)
    rw_2=0        # for 2 in a row
    rw_2enemy=0   # for 2 in a row
    rws_init=[0,  0.3,  0.5,  0.7,  0.4,  0.2,  0] # initial rewards (prefer center slightly)
    
    # me:me_or_enemy=1, enemy:me_or_enemy=2
    def check_vertical_chance(me_or_enemy,board=[], n=4):
        if len(board)==0:
            board=observation.board
        chances=[]
        for i in range(0, 7):
            for j in range(6-n+1,6):
                for k in range(n-1): # check if vertical row for te palyer and empty last cell
                    if board[i+7*(j-k)] != me_or_enemy: # if all same player
                        break
                    if k==n-2 and board[i+7*(j-k-1)] == 0: # last cell empty
                        chances.append(i)
        return chances
            
    # me:me_or_enemy=1, enemy:me_or_enemy=2
    def check_horizontal_chance(me_or_enemy,board=[], n=4):
        if len(board)==0:
            board=observation.board
        chances=[]
        for i in range(6):
            for j in range(0, 7-n+1): 
                sums=sum([board[i*7 + j+s] == me_or_enemy for s in range(n)])
                if sums == n-1:
                    for k in [i*7 + j+s for s in range(n)]:
                        if board[k] == 0:
                            chance_cell_num = k
                            # bottom line
                            if chance_cell_num in range(35, 42):
                                chances.append(chance_cell_num - 35)
                            # others
                            elif board[chance_cell_num+7] != 0:
                                chances.append(chance_cell_num % 7)
        return chances            
    
    # me:me_or_enemy=1, enemy:me_or_enemy=2
    def check_slanting_chance(me_or_enemy, lag, cell_list,board=[], n=4):
        if len(board)==0:
            board=observation.board
        chances=[]
        for i in cell_list:
            sums=sum([board[i+lag*s] == me_or_enemy for s in range(n)])
            if sums == n-1:
                for j in [i+lag*s for s in range(n)]:
                    if board[j] == 0:
                        chance_cell_num = j
                        # bottom line
                        if chance_cell_num in range(35, 42):
                            chances.append(chance_cell_num - 35)
                        # others
                        elif board[chance_cell_num+7] != 0:
                            chances.append(chance_cell_num % 7)
        return chances
    
    def check_horizontal_first_enemy_chance():
        # enemy's chance
        if observation.board[38] == enemy_num:
            if sum([observation.board[39] == enemy_num, observation.board[40] == enemy_num]) == 1             and observation.board[37] == 0:
                for i in range(39, 41):
                    if observation.board[i] == 0:
                        return i - 35
            if sum([observation.board[36] == enemy_num, observation.board[37] == enemy_num]) == 1             and observation.board[39] == 0:
                for i in range(36, 38):
                    if observation.board[i] == 0:
                        return i - 35
        return -99 # no chance

    def check_first_or_second():
        count = 0
        for i in observation.board:
            if i != 0:
                count += 1
        # first
        if count % 2 != 1:
            my_num = 1
            enemy_num = 2
        # second
        else:
            my_num = 2
            enemy_num = 1
        return my_num, enemy_num
        
    def check_my_chances():
        # check my virtical chance
        result = check_vertical_chance(my_num)
        if len(result)>0:
            return result[0]
        # check my horizontal chance
        result = check_horizontal_chance(my_num)
        if len(result)>0:
            return result[0]
        # check my slanting chance 1 (up-right to down-left)
        result = check_slanting_chance(my_num, 6, [3,4,5,6,10,11,12,13,17,18,19,20])
        if len(result)>0:
            return result[0]
        # check my slanting chance 2 (up-left to down-right)
        result = check_slanting_chance(my_num, 8, [0,1,2,3,7,8,9,10,14,15,16,17])
        if len(result)>0:
            return result[0]
        # no chance
        return -99
    
    def check_enemy_chances():
        # check horizontal first chance
        result = check_horizontal_first_enemy_chance()
        if result != -99:
            return result
        # check enemy's vertical chance
        result = check_vertical_chance(enemy_num)
        if len(result)>0:
            return result[0]
        # check enemy's horizontal chance
        result = check_horizontal_chance(enemy_num)
        if len(result)>0:
            return result[0]
        # check enemy's slanting chance 1 (up-right to down-left)
        result = check_slanting_chance(enemy_num, 6, [3,4,5,6,10,11,12,13,17,18,19,20])
        if len(result)>0:
            return result[0]
        # check enemy's slanting chance 2 (up-left to down-right)
        result = check_slanting_chance(enemy_num, 8, [0,1,2,3,7,8,9,10,14,15,16,17])
        if len(result)>0:
            return result[0]
        # no chance
        return -99

    # check first or second
    my_num, enemy_num = check_first_or_second()

    # defines the placed cell for playing pos. ex: playing 0 could place cell 14
    def RealPos(pos,board):
        real_pos=None
        for d in reversed(range(6)):
            if board[pos+7*d]==0:
                real_pos=pos+7*d
                break
        return real_pos
    
    # simulate play one step and get possible rewards
    def sim_play(player=my_num, opponent=enemy_num, board=[]):
        # check real position when playing
        # check max lengths as reward (non-blocked lengths)
        # choose max reward     
        rws=rws_init.copy() # # use priority - 3 > 2 > 4 > 1 > 5 > 0 > 6
        if len(board)==0:
            board=observation.board.copy()
        for i in range(7):
            col_empty=[j for j in range(6) if board[i+7*j]==0]
            if len(col_empty)==0:
                rws[i]=-9999
            else:
                pos=max(col_empty)
                board2=board.copy()
                board2[i+pos*7]=player

                # makin 4 in a row chanes
                v=check_vertical_chance(player,board2)
                h=check_horizontal_chance(player,board2)
                s1=check_slanting_chance(player, 6, [3,4,5,6,10,11,12,13,17,18,19,20],board2)
                s2=check_slanting_chance(player, 8, [0,1,2,3,7,8,9,10,14,15,16,17],board2)
                chances = v + h + s1 + s2
                rws[i]+=len(chances)**2 * rw_win
                # making 3 in a row chances
                v=check_vertical_chance(player,board2,n=3)
                h=check_horizontal_chance(player,board2,n=3)
                s1=check_slanting_chance(player, 6, [2,3,4,5,9,10,11,12,16,17,18,19,23,24,25,26],board2,n=3)
                s2=check_slanting_chance(player, 8, [1,2,3,4,8,9,10,11,15,16,17,18,22,23,24,25],board2,n=3)
                chances = v + h + s1 + s2
                rws[i]+=len(chances)**2 * rw_3
                # making 2 in a row chances
                rws[i]+=len(chances) * rw_win
                v=check_vertical_chance(player,board2,n=2)
                h=check_horizontal_chance(player,board2,n=2)
                s1=check_slanting_chance(player, 6, [1,2,3,4, 8,9,10,11, 15,16,17,18, 22,23,24,25],board2,n=2)
                s2=check_slanting_chance(player, 8, [2,3,4,5, 9,10,11,12, 16,17,18,19, 23,24,25,26],board2,n=2)
                chances = v + h + s1 + s2
                rws[i]+=len(chances) * rw_2

                # making 4 in a row chanes for opponent
                v=check_vertical_chance(opponent,board2)
                h=check_horizontal_chance(opponent,board2)
                s1=check_slanting_chance(opponent, 6, [3,4,5,6,10,11,12,13,17,18,19,20],board2)
                s2=check_slanting_chance(opponent, 8, [0,1,2,3,7,8,9,10,14,15,16,17],board2)
                chances = v + h + s1 + s2
                rws[i]+=len(chances)**2 * rw_loss
                # making 3 in a row chanes for opponent
                v=check_vertical_chance(opponent,board2,n=3)
                h=check_horizontal_chance(opponent,board2,n=3)
                s1=check_slanting_chance(opponent, 6, [2,3,4,5,9,10,11,12,16,17,18,19,23,24,25,26],board2,n=3)
                s2=check_slanting_chance(opponent, 8, [1,2,3,4,8,9,10,11,15,16,17,18,22,23,24,25],board2,n=3)
                chances = v + h + s1 + s2
                rws[i]+=len(chances)**2 * rw_3enemy
                # making 2 in a row chanes for opponent
                v=check_vertical_chance(opponent,board2,n=2)
                h=check_horizontal_chance(opponent,board2,n=2)
                s1=check_slanting_chance(opponent, 6, [1,2,3,4, 8,9,10,11, 15,16,17,18, 22,23,24,25],board2,n=2)
                s2=check_slanting_chance(opponent, 8, [2,3,4,5, 9,10,11,12, 16,17,18,19, 23,24,25,26],board2,n=2)
                chances = v + h + s1 + s2
                rws[i]+=len(chances) * rw_2enemy

        return rws
    
    def play_deep(alpha=0.6):
        rws=sim_play()
        #rws=np.array(rws)
        #moves=rws.argsort()[-4:][::-1]
        board=observation.board.copy()
        for move in range(7): # check all 7 moves
            board2=board.copy()
            real_pos=RealPos(move,board2)
            if real_pos==None: # move is not posibble
                continue
            board2[real_pos]=my_num
            rws_enemy=sim_play(player=enemy_num, opponent=my_num, board=board2)
            rws_enemy_=np.array(rws_enemy)
            moves2=[t for t in range(7)] # check all 7 moves
            if rws_enemy!=rws_init:
                rws[move] -= alpha * sum(sorted(rws_enemy)[-3:])/3 # - alpha * average of best 3 move
                moves2=rws_enemy_.argsort()[-3:][::-1] # best 3 enemy move
            for move2 in moves2:
                board3=board2.copy()
                real_pos=RealPos(move2,board3)
                if real_pos==None:
                    continue                
                board3[real_pos]=enemy_num
                rws_me=sim_play(player=my_num, opponent=enemy_num, board=board3)
                rws[move] += alpha*alpha * sum(sorted(rws_me)[-3:])/3 # + alpha^2 * average of best 3 move
        best_rws=-9999
        best_move=0
        for move in range(7):
            if rws[move]>best_rws:
                best_move=move
        return int(best_move)
    
    # play for the next step not further ahead
    def play_greedy():
        rws=sim_play()
        move=int(np.argmax(rws))
        return move

    # if immediate no immediate reward, calculate deeper
    def play_partial():
        rws=sim_play()
        if rws!=rws_init:
            move=int(np.argmax(rws))
            return move
        else:
            move = play_deep()
            return move
        move=int(np.argmax(rws))
        return move
    
    #################################################
    ####################  PLAY  #####################
    #################################################
    
    # check immediate wins and play if occurs
    result = check_my_chances()
    if result != -99:
        return result
    
    # check possible immediate lost and prevent
    result = check_enemy_chances()
    if result != -99:
        return result
    
    # decision for no immediate win or loss
    move=play_greedy()
    return move


# # Test your Agent

# In[ ]:


env.reset()
# Play as the first agent against default "random" agent.
#env.run([my_agent, "random"])
env.run([agent, "negamax"])
env.render(mode="ipython", width=300, height=300)


# In[ ]:


env.reset()
# Play as the first agent against default "random" agent.
env.run(["random", agent])
# env.run([agent, "negamax"])
env.render(mode="ipython", width=300, height=300)


# In[ ]:


env.reset()
# my agent VS my agent
env.run([agent, agent])
# env.run([agent, "negamax"])
env.render(mode="ipython", width=300, height=300)


# # Debug/Train your Agent

# In[ ]:


# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()


# In[ ]:


while not env.done:
    my_action = agent(observation, env.configuration)
    print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
    env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()


# # Evaluate your Agent

# In[ ]:


def mean_reward(rewards):
    win=sum(r[0] for r in rewards if r[0] not in [None,0.5])
    loss=sum(r[1] for r in rewards if r[1] not in [None,0.5])
    eq=len(rewards)-win-loss
    rew= (win+eq/2) / (win+loss+eq)
    rew=int(rew*1000+0.5)/1000.0
    return rew, win, loss, eq

print("My Agent vs My Agent:", mean_reward(evaluate("connectx", [agent, agent], num_episodes=1)))

"""
----------------- 0 -----------------
Random vs Agent: (0.003, 3, 997, 0)
Agent vs Random: (0.999, 998, 1, 1)
 
----------------- 1 -----------------
Random vs Agent: (0.004, 3, 996, 1)
Agent vs Random: (1.0, 1000, 0, 0)
 
----------------- 2 -----------------
Random vs Agent: (0.001, 1, 999, 0)
Agent vs Random: (0.998, 998, 2, 0)
"""
# Run multiple episodes to estimate it's performance.
# remove for commit to save time, increase repeat and runs for real performance test
repeat=3
runs=10
for i in range(repeat):
    print("-----------------",i,"-----------------")
    print("Random vs Agent:", mean_reward(evaluate("connectx", ["random", agent], num_episodes=runs)))
    print("Agent vs Random:", mean_reward(evaluate("connectx", [agent, "random"], num_episodes=runs)))
    print(" ")


# # Write Submission File
# 
# 

# In[ ]:


import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(agent, "submission.py")
print("Done")

