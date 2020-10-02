#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# This Monte Carlo algorithm improves a simple lookahead by running as many simulated games from the current position as possible within our turn time limit. We choose the move that appears to have the greatest chance of victory based on the simulations. 

# In[ ]:


from kaggle_environments import make, evaluate

# Create the game environment
# Set debug=True to see the errors if your agent refuses to run
env = make("connectx", debug=True)


# Agents are functions that take two parameters:
# 
# - `obs.board` : the game board (a Python list with one item for each grid location)
# - `obs.mark` : the piece assigned to the agent (either 1 or 2)
# 
# - `config.columns` : number of columns in the game board (7 for Connect Four)
# - `config.rows` : number of rows in the game board (6 for Connect Four)
# - `config.inarow` : number of pieces a player needs to get in a row in order to win (4 for Connect Four)
# 
# Agents return the column to drop their next piece into (0-6).
# 
# Remember that the agent function must be completely stand-alone! It must contain *all* helper functions and imports.

# In[ ]:


def agent_monte_carlo(obs, config):
    import random
    import time
    import numpy as np
    import math

    #############
    ## Helpers ##
    #############
    def board_to_grid(board, config):
        return np.asarray(board).reshape(config.rows, config.columns)

    def grid_to_board(grid):
        return grid.reshape(-1)

    def other_player(player):
        return 1 if player == 2 else 2 

    # Gets grid at next step if agent drops piece in selected column
    def drop_piece(grid, col, piece, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = piece
        return next_grid, row

    # The "4" at the end of the name idicates this function only works when dealing with 
    # games that terminate with 4 in-a-row.
    # This version checks the full board; the one below checks only the spaces affected by the 
    # latest drop.
    def player_has_won_fast_4_full(grid, config):
        assert config.inarow == 4

        for r in range(config.rows):
            for c in range(config.columns-3):
                if 0 != grid[r][c] == grid[r][c+1] == grid[r][c+2] == grid[r][c+3]:
                    return grid[r][c]

        for c in range(config.columns):
            for r in range(config.rows-3):
                if 0 != grid[r][c] == grid[r+1][c] == grid[r+2][c] == grid[r+3][c]:
                    return grid[r][c]

        for r in range(config.rows-3):
            for c in range(config.columns-3):
                if 0 != grid[r][c] == grid[r+1][c+1] == grid[r+2][c+2] == grid[r+3][c+3]:
                    return grid[r][c]

        for r in range(config.rows-3):
            for c in range(config.columns-3):
                if 0 != grid[r][c+3] == grid[r+1][c+2] == grid[r+2][c+1] == grid[r+3][c]:
                    return grid[r][c+3]

        return 0
    
    def player_has_won_fast_4(grid, row, col, config):
        assert config.inarow == 4

        # Check horizontal
        for c in range(max(0, col-3), min(col, config.columns-3)):
            if 0 != grid[row][c] == grid[row][c+1] == grid[row][c+2] == grid[row][c+3]:
                return grid[row][col]

        # Check vertical
        for r in range(max(0, row-3), min(row, config.rows-3)):
            if 0 != grid[r][col] == grid[r+1][col] == grid[r+2][col] == grid[r+3][col]:
                return grid[row][col]

        # Check diagonal down-right
        for i in range(max(-3, -row, -col), min(0, -3+config.rows-1-row, -3+config.columns-1-col)):
            if 0 != grid[row+i][col+i] == grid[row+i+1][col+i+1] == grid[row+i+2][col+i+2] == grid[row+i+3][col+i+3]:
                return grid[row][col]

        # Check diagonal down-left
        for i in range(max(-3, -row, -config.columns+1+col), min(0, -3+config.rows-1-row, -3+col)):
            if 0 != grid[row+i][col-i] == grid[row+i+1][col-i-1] == grid[row+i+2][col-i-2] == grid[row+i+3][col-i-3]:
                return grid[row][col]

        return 0

    # This function defines a simple rule-based look-ahead agent. 
    # Returns (col, guaranteed_win)
    def behavior_lookahead_2(grid, piece, config):
        valid_moves = [col for col in range(config.columns) if grid[0][col] == 0]

        if len(valid_moves) == 0:
            return None, False

        # If dropping a piece makes us win, then do that.
        for move in valid_moves:   
            next_grid, row = drop_piece(grid, move, piece, config)
            if player_has_won_fast_4(next_grid, row, move, config) != 0:
                return move, True

        # If dropping a piece blocks our opponent from winning next turn, then do that.
        for move in valid_moves:    
            next_grid, row = drop_piece(grid, move, other_player(piece), config)
            if player_has_won_fast_4(next_grid, row, move, config) != 0:
                return move, False

        # If dropping a piece gives us two ways to win next turn, then do that.
#         for move_1 in valid_moves:
#             paths_to_victory = 0
#             next_grid, _ = drop_piece(grid, move_1, piece, config)
#             next_valid_moves = [col for col in range(config.columns) if next_grid[0][col] == 0]
#             for move_2 in next_valid_moves:
#                 next_grid_2, row = drop_piece(next_grid, move_2, piece, config)
#                 if player_has_won_fast_4(next_grid_2, row, move, config) != 0:
#                     paths_to_victory += 1
#                     if paths_to_victory >= 2:
#                         return move_1, True
        
        # Otherwise, choose a random valid move
        return random.choice(valid_moves), False
    
    discount_factor = 0.9
    max_depth = 10

    # Simulate two lookahead_2 players from the given grid position.
    def simulate(move, grid, row, col):
        us = obs.mark
        them = other_player(obs.mark)

        if player_has_won_fast_4(grid, row, col, config) == us:
            return 1.0

        next_grid = grid
        k = 0

        while time.time() < deadline:
            their_move, guaranteed_win = behavior_lookahead_2(next_grid, them, config)
            if their_move == None:
                return 0
            if guaranteed_win:
                # They won
                return -1.0 * discount_factor**k      
            next_grid, _ = drop_piece(next_grid, their_move, them, config)
            k+=1

            our_move, guaranteed_win = behavior_lookahead_2(next_grid, us, config)
            if our_move == None:
                return 0
            if guaranteed_win:
                # We won
                return 1.0 * discount_factor**k
            next_grid, _ = drop_piece(next_grid, our_move, us, config)
            k+=1
            
            if k >= max_depth:
                return "max_depth"
        
        return "timeup"

    def choose_next_move_uniform(valid_moves):
        return random.choice(valid_moves)
    
    ###########
    ## AGENT ##
    ###########
    
    deadline = time.time() + config.actTimeout - 0.5
    # Uncomment to limit time per turn during testing.
    # deadline = time.time() + 1
    
    grid = board_to_grid(obs.board, config)
    
    valid_moves = [col for col in range(config.columns) if grid[0][col] == 0]
    
    # Initialize all values to -2. Invalid moves will never be updated, so they will always be
    # worse than moves that guarantee defeat. This prevents us from erroring out.
    values = np.repeat(-2.0, config.columns)
    
    num_observed = np.zeros_like(values)
    
    while time.time() < deadline:
        move = choose_next_move_uniform(valid_moves)
        
        # Estimate return
        next_grid, row = drop_piece(grid, move, obs.mark, config)
        _return = simulate(move, next_grid, row, move)
        if _return == "timeup":
            # Ended early. Time is almost up!
            break
        if _return == "max_depth":
            continue
        
        # Update estimated values
        num_observed[move] += 1
        values[move] = _return / num_observed[move] +                        (num_observed[move] - 1) / num_observed[move] * values[move] 
    
    return int(np.argmax(values))


# In[ ]:


# Copy the helper methods from the agent_monte_carlo function to try this out.

# g_board = None
# g_config = None

# def agent_lookahead_2(obs, config):
#     global g_board, g_config
#     grid = board_to_grid(obs.board, config)
#     g_board = obs.board
#     g_config = config
#     move, _ behavior_lookahead_2(grid, obs.mark, config)
#     return move


# In[ ]:


# # Two agents play one game round
# env.run([agent_monte_carlo, "random"]);
# # Show the game
# env.render(mode="ipython")


# In[ ]:


# grid = board_to_grid(g_board, g_config)

# %timeit drop_piece(grid, 4, 1, g_config)


# In[ ]:


# Modified from tutorial

# def get_win_percentages(agent1, agent2, n_rounds=50):
#     import numpy as np
#     config = {"rows": 6, "columns": 7, "inarow": 4}
#     outcomes = []
#     for i in range(n_rounds):
#         outcomes += evaluate("connectx", [agent1, agent2], config, [], 1)
#         outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], 1)]
#     print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
#     print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
#     return outcomes
    
# outcomes = get_win_percentages(agent_lookahead_2, agent_lookahead_3, 50)


# In[ ]:


import inspect
import os

# Create submission file
def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)
        
write_agent_to_file(agent_monte_carlo, "submission.py")


# In[ ]:


# Validate submission file

import sys
from kaggle_environments import utils

out = sys.stdout
submission = utils.read_file("/kaggle/working/submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")


# To sumit to the competition, follow these steps:
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
