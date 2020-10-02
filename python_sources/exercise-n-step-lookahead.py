#!/usr/bin/env python
# coding: utf-8

# **[Intro to Game AI and Reinforcement Learning Home Page](https://www.kaggle.com/learn/intro-to-game-ai-and-reinforcement-learning)**
# 
# ---
# 

# # Introduction
# 
# In the tutorial, you learned how to build a reasonably intelligent agent with the minimax algorithm.  In this exercise, you will check your understanding and submit your own agent to the competition.

# In[ ]:


from learntools.core import binder
binder.bind(globals())
from learntools.game_ai.ex3 import *


# ### 1) A closer look
# 
# The heuristic from the tutorial looks at all groups of four adjacent grid locations on the same row, column, or diagonal and assigns points for each occurrence of the following patterns:
# 
# <center>
# <img src="https://i.imgur.com/3NvBEGL.png" width=70%><br/>
# </center>
# 
# Is it really necessary to use so many numbers to define the heuristic?  Consider simplifying it, as in the image below.
# 
# <center>
# <img src="https://i.imgur.com/grViegG.png" width=70%><br/>
# </center>
# 
# How would each heuristic score the potential moves in the example below (where, in this case, the agent looks only one step ahead)?  Which heuristic would lead to the agent selecting the better move?
# 
# <center>
# <img src="https://i.imgur.com/LWPLy7N.png" width=100%><br/>
# </center>

# In[ ]:


"""
-1, -1, 0, 0(-1+1/will lose), -1, -1, -1
-1000+100, -1000+100, 100(best), -1000+1000, -1000+100+100, -1000+100+100, -1000+100+100
"""


# In[ ]:


q_1.hint()


# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_1.solution()


# ### 2) Count the leaves
# 
# In the tutorial, we worked with a small game tree.
# 
# <center>
# <img src="https://i.imgur.com/BrRe7Bu.png" width=90%><br/>
# </center>
# 
# The game tree above has 8 leaf nodes that appear at the bottom of the tree.  By definition, "leaf nodes" in a game tree are nodes that don't have nodes below them.
# 
# In the ConnectX competition, the game trees will be much larger!  
# 
# To see this, consider a minimax agent that is trying to plan its first move, where all columns in the game board are  empty.  Say the agent builds a game tree of depth 3.  How many leaf nodes are in the game tree?  
# 
# Use your answer to fill in the blank below.

# In[ ]:


2**3


# In[ ]:


# Fill in the blank
num_leaves = 7**3

# Check your answer
q_2.check()


# In[ ]:


num_leaves


# In[ ]:


# Lines below will give you a hint or solution code
#q_2.hint()
q_2.solution()


# ### 3) Which move will the agent select?
# 
# In this question, you'll check your understanding of the minimax algorithm.  Remember that with this algorithm, 
# > The agent chooses moves to get a score that is as high as possible, and it assumes the opponent will counteract this by choosing moves to force the score to be as low as possible.
# 
# Consider the toy example below of a game tree that the agent will use to select its next move.  
# <center>
# <img src="https://i.imgur.com/QlfWGM9.png" width=80%><br/>
# </center>
# 
# Which move will the agent select?  Use your answer to set the value of the `selected_move` variable below.  Your answer should be one of `1`, `2`, or `3`.

# In[ ]:


-100, -80, 1


# In[ ]:


# Fill in the blank
selected_move = 3

# Check your answer
q_3.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_3.hint()
q_3.solution()


# ### 4) Examine the assumptions
# 
# The minimax agent assumes that its opponent plays optimally (with respect to the heuristic, and using a game tree of limited depth).  But this is almost never the case, in practice: it's far more likely for the agent to encounter a suboptimal (that is: worse than optimal) opponent.  
# 
# Say the minimax agent encounters a suboptimal opponent. Should we expect the minimax agent to still play the game well, despite the contradiction with its assumptions?  If so, why?

# In[ ]:


q_4.hint()


# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_4.solution()


# ### 5) Submit to the competition
# 
# Now, it's time to submit an agent to the competition!  Use the next code cell to define an agent.  (You can see an example of how to write a valid agent in **[this notebook](https://www.kaggle.com/alexisbcook/create-a-connectx-agent)**.)
# 
# If you decide to use the minimax code from the tutorial, you might like to add [**alpha-beta pruning**](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) to decrease the computation time (i.e., get the minimax algorithm to run much faster!).  In this case, "alpha" and "beta" to refer to two values that are maintained while the algorithm is running, that help to identify early stopping conditions.  
# 
# Without alpha-beta pruning, minimax evaluates each leaf node.  With alpha-beta pruning, minimax only evaluates nodes that could provide information that affects the agent's choice of action.  Put another way, it identifies nodes that could not possibly affect the final result and avoids evaluating them.

# In[ ]:


def agent_alphabeta(obs, config):

    ################################
    # Imports and helper functions #
    ################################
    
    import numpy as np
    import random

    # Helper function for is_terminal_node: checks if agent or opponent has four in a row in the window
    def is_terminal_window(window, config):
        return window.count(1) == config.inarow or window.count(2) == config.inarow

    # Helper function for minimax_alphabeta: checks if game has ended
    def is_terminal_node(grid, config):
        # Check for draw 
        if list(grid[0, :]).count(0) == 0:
            return True
        # Check for win: horizontal, vertical, or diagonal
        # horizontal 
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if is_terminal_window(window, config):
                    return True
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if is_terminal_window(window, config):
                    return True
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        return False

    # Helper function for count_windows: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)

    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        return num_windows

    # Helper function for minimax: calculates value of heuristic for grid
    def get_heuristic(grid, mark, config):
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_threes_opp = count_windows(grid, 3, mark%2+1, config)
        num_fours_opp = count_windows(grid, 4, mark%2+1, config)
        score = num_threes - 1e2*num_threes_opp - 1e4*num_fours_opp + 1e6*num_fours
        return score

    # Gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    # Minimax implementation: recursive!!!
    def minimax_alphabeta(node, depth, alpha, beta, maximizingPlayer, mark, config):
        is_terminal = is_terminal_node(node, config)
        valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
        if depth == 0 or is_terminal:
            return get_heuristic(node, mark, config)
        if maximizingPlayer:
            value = -np.Inf
            for col in valid_moves:
                child = drop_piece(node, col, mark, config)
                value = max(value, minimax_alphabeta(child, depth-1, alpha, beta, False, mark, config))
                alpha = max(alpha, value) # the minimum score that the maximizing player (i.e., the "alpha" player) is assured of
                if alpha >= beta: #
                    break # beta cutoff
            return value
        else:
            value = np.Inf
            for col in valid_moves:
                child = drop_piece(node, col, mark%2+1, config)
                value = min(value, minimax_alphabeta(child, depth-1, alpha, beta, True, mark, config))
                beta = min(beta, value) # the maximum score that the minimizing player (i.e. the "beta" player) is assured of
                if beta <= alpha: #
                    break # alpha cutoff
            return value

    # Uses minimax to calculate value of dropping piece in selected column
    def score_move(grid, col, mark, config, nsteps):
        return minimax_alphabeta(grid, nsteps, -np.Inf, np.Inf, True, mark, config)

    #########################
    # Agent makes selection #
    #########################

    # How deep to make the game tree: higher values take longer to run!
    N_STEPS = 3

    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)


# In[ ]:


# Run this code cell to get credit for creating an agent
q_5.check()


# In[ ]:


import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(agent_alphabeta, "submission.py")


# In[ ]:


from kaggle_environments import make, evaluate

# Create the game environment
# Set debug=True to see the errors if your agent refuses to run
env = make("connectx", debug=True)


# In[ ]:


import sys
from kaggle_environments import utils

out = sys.stdout
submission = utils.read_file("/kaggle/working/submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")


# In[ ]:


# Two random agents play one game round
env.run([agent_alphabeta, "random"])

# Show the game
env.render(mode="ipython")


# In[ ]:


env.play([agent_alphabeta, None], width=500, height=450)


# Then, follow these steps to submit your agent to the competition:
# 1. Begin by clicking on the blue **Save Version** button in the top right corner of this window.  This will generate a pop-up window.  
# 2. Ensure that the **Save and Run All** option is selected, and then click on the blue **Save** button.
# 3. This generates a window in the bottom left corner of the notebook.  After it has finished running, click on the number to the right of the **Save Version** button.  This pulls up a list of versions on the right of the screen.  Click on the ellipsis **(...)** to the right of the most recent version, and select **Open in Viewer**.  This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.
# 4. Click on the **Output** tab on the right of the screen.  Then, click on the **Submit to Competition** button to submit your results to the leaderboard.
# 
# You have now successfully submitted to the competition!
# 
# If you want to keep working to improve your performance, select the blue **Edit** button in the top right of the screen. Then you can change your code and repeat the process. There's a lot of room to improve, and you will climb up the leaderboard as you work.
# 
# 
# Go to **"My Submissions"** to view your score and episodes being played.
# 
# # Keep going
# 
# Move on to learn how to **[use deep reinforcement learning](https://www.kaggle.com/alexisbcook/deep-reinforcement-learning)** to develop an agent without a heuristic!

# ---
# **[Intro to Game AI and Reinforcement Learning Home Page](https://www.kaggle.com/learn/intro-to-game-ai-and-reinforcement-learning)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
