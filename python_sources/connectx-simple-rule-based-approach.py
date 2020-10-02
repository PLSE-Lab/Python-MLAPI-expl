#!/usr/bin/env python
# coding: utf-8

# # Install kaggle-environments

# In[ ]:


# 1. Enable Internet in the Kernel (Settings side pane)

# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 
# !curl -X PURGE https://pypi.org/simple/kaggle-environments

# ConnectX environment was defined in v0.1.6
get_ipython().system("pip install 'kaggle-environments>=0.1.6'")


# # Create ConnectX Environment

# In[ ]:


from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)
env.render()


# # Create an Agent
# 
# To create the submission, an agent function should be fully encapsulated (no external dependencies).  
# 
# When your agent is being evaluated against others, it will not have access to the Kaggle docker image.  Only the following can be imported: Python Standard Library Modules, gym, numpy, scipy, pytorch (1.3.1, cpu only), and more may be added later.
# 
# 

# In[ ]:


def my_agent(observation, configuration):
    PLAYER = observation.mark
    OPPONENT = 3 - PLAYER

    def make_move(board, move, player):
        for i in range(5, -1, -1):
            new_piece = move + 7*i
            if board[new_piece] == 0:
                board[new_piece] = player
                return board, new_piece
        return None, None # Illegal move

    def check_win(board, move, player):
        if board[move] != 0: # Full Column
            return False

        _, new_piece = make_move(board, move, player)
        # check horizontal spaces
        for j in range(4):
            if new_piece + j > 41:
                break
            if (new_piece + j) % 7 < 3:
                continue
            if board[new_piece + j] == player and board[new_piece + j - 1] == player and board[new_piece + j - 2] == player and board[new_piece + j - 3] == player:
                return True

        # check vertical spaces
        for j in range(4):
            if new_piece + j*7 > 41:
                break
            if new_piece +j*7 < 21:
                continue
            if board[new_piece + j*7] == player and board[new_piece + j*7 - 7] == player and board[new_piece + j*7 - 14] == player and board[new_piece + j*7 - 21] == player:
                return True

        # check diagonal descending spaces
        for j in range(4):
            if new_piece + j*8 > 41:
                break
            if new_piece + j*8 < 24 or (new_piece + j*8) % 7 < 3:
                continue
            if board[new_piece + j*8] == player and board[new_piece + j*8 - 8] == player and board[new_piece + j*8 - 16] == player and board[new_piece + j*8 - 24] == player:
                return True

        # check diagonal ascending spaces
        for j in range(4):
            if new_piece + j*6 > 41:
                break
            if (new_piece + j*6) % 7 > 3 or new_piece + j*6 < 21:
                continue
            if board[new_piece + j*6] == player and board[new_piece + j*6 - 6] == player and board[new_piece + j*6 - 12] == player and board[new_piece + j*6 - 18] == player:
                return True

        return False
    
    move_order = [3, 2, 4, 1, 5, 0, 6]
    for i in range(configuration.columns):
        if check_win(observation.board.copy(), i, PLAYER):
            return i
        else:
            for j in range(configuration.columns):
                new_board, _ = make_move(observation.board.copy(), i, PLAYER)
                if new_board is not None and i in move_order and check_win(new_board.copy(), j, OPPONENT):
                    move_order.remove(i)


    for i in range(configuration.columns):
        if check_win(observation.board.copy(), i, OPPONENT):
            return i
    
    # Hardcoding the defense to an early game weakness
    if observation.board[38] == OPPONENT:
        if observation.board[37] == OPPONENT and observation.board[39] == 0:
            return 4
        elif observation.board[39] == OPPONENT and observation.board[37] == 0:
            return 2
    
    assert move_order is not None
    for i in move_order:
        if observation.board[i] == 0:
            return i
    
    # dead end
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])


# # Test your Agent

# In[ ]:


env.reset()
# Play as the first agent against default "random" agent.
env.run([my_agent, "random"])
env.render(mode="ipython", width=500, height=450)


# # Debug/Train your Agent

# In[ ]:


# Play as first position against random agent.
trainer = env.train([None, "random"])
observation = trainer.reset()

while not env.done:
    my_action = my_agent(observation, env.configuration)
    print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
    print(observation, reward)
    # env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()


# # Evaluate your Agent

# In[ ]:


def mean_reward(rewards):
    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)

# Run multiple episodes to estimate its performance.
print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=1000)))
print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=1)))


# # Play your Agent
# Click on any column to place a checker there ("manually select action").

# In[ ]:


# "None" represents which agent you'll manually play as (first or second player).
env.play([None, my_agent], width=500, height=450)


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

write_agent_to_file(my_agent, "submission.py")


# # Validate Submission
# Play your submission against itself.  This is the first episode the competition will run to weed out erroneous agents.
# 
# Why validate? This roughly verifies that your submission is fully encapsulated and can be run remotely.

# In[ ]:


# Note: Stdout replacement is a temporary workaround.
import sys
out = sys.stdout
submission = utils.read_file("/kaggle/working/submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")


# # Submit to Competition
# 
# 1. Commit this kernel.
# 2. View the commited version.
# 3. Go to "Data" section and find submission.py file.
# 4. Click "Submit to Competition"
# 5. Go to [My Submissions](https://kaggle.com/c/connectx/submissions) to view your score and episodes being played.
