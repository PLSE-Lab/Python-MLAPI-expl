#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from kaggle_environments import make, evaluate

env = make("connectx", debug=True)


# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', '\nimport random\nimport time\nimport numpy as np\nfrom collections import defaultdict\n\nROWS = 6\nCOLUMNS = 7\nINAROW = 4 \n\n# Params\ndiscount_factor = 0.9\nlr = 0.1\n\ndef board_to_grid(board, config):\n    return np.asarray(board).reshape(ROWS, COLUMNS)\n\n\ndef grid_to_board(grid):\n    return grid.reshape(-1)\n\n\ndef other_player(player):\n    return 1 if player == 2 else 2 \n\n# Zobrist hash \nHASH_TABLE = np.frombuffer(\n    np.random.bytes(ROWS*COLUMNS*3*8), dtype=np.int64\n).reshape([ROWS*COLUMNS,3])\n\ndef hash_board(board):\n    return np.bitwise_xor.reduce(HASH_TABLE[np.arange(ROWS*COLUMNS), board])\n\n# Value estimates\nV = defaultdict(float)\n\ndef get_value(board):\n    return V[hash_board(board)]\n\n\ndef set_value(board, val):\n    V[hash_board(board)] = val\n\n    \ndef update_estimate(board, target):\n    v = get_value(board)\n    set_value(board, v + lr * (target - v))\n    \n    \n# Gets grid at next step if agent drops piece in selected column\ndef drop_piece(grid, col, piece, config):\n    next_grid = grid.copy()\n    for row in range(config.rows-1, -1, -1):\n        if next_grid[row][col] == 0:\n            break\n    next_grid[row][col] = piece\n    return next_grid\n\n\n# The "4" at the end of the name idicates this function only works when dealing with \n# games that terminate with 4 in-a-row.\ndef player_has_won_fast_4(grid, config):\n    assert config.inarow == 4\n\n    for r in range(config.rows):\n        for c in range(config.columns-3):\n            if 0 != grid[r][c] == grid[r][c+1] == grid[r][c+2] == grid[r][c+3]:\n                return grid[r][c]\n\n    for c in range(config.columns):\n        for r in range(config.rows-3):\n            if 0 != grid[r][c] == grid[r+1][c] == grid[r+2][c] == grid[r+3][c]:\n                return grid[r][c]\n\n    for r in range(config.rows-3):\n        for c in range(config.columns-3):\n            if 0 != grid[r][c] == grid[r+1][c+1] == grid[r+2][c+2] == grid[r+3][c+3]:\n                return grid[r][c]\n\n    for r in range(config.rows-3):\n        for c in range(config.columns-3):\n            if 0 != grid[r][c+3] == grid[r+1][c+2] == grid[r+2][c+1] == grid[r+3][c]:\n                return grid[r][c+3]\n\n    return 0\n\n\ndef behavior_lookahead_1(grid, piece, config):\n    valid_moves = [col for col in range(config.columns) if grid[0][col] == 0]\n\n    if len(valid_moves) == 0:\n        return None\n\n    # If dropping a piece makes us win, then do that.\n    for move in valid_moves:   \n        next_grid = drop_piece(grid, move, piece, config)\n        if player_has_won_fast_4(next_grid, config) != 0:\n            return move\n\n    # If dropping a piece blocks our opponent from winning next turn, then do that.\n    for move in valid_moves:    \n        next_grid = drop_piece(grid, move, other_player(piece), config)\n        if player_has_won_fast_4(next_grid, config) != 0:\n            return move\n\n    # Otherwise, choose a random valid move\n    return random.choice(valid_moves)\n\n\n# Simulate two lookahead_1 players from the given grid position.\ndef simulate(move, player, grid, obs, config):\n    \n    next_grid = drop_piece(grid, move, player, config)\n    \n    winner = player_has_won_fast_4(next_grid, config)\n    \n    if winner == obs.mark:\n        # Us\n        reward = 1.0\n    elif winner != 0:\n        # Them\n        reward = -1.0\n    else:\n        # Neither, keep simulating\n        next_player = other_player(player)\n        next_move = behavior_lookahead_1(next_grid, next_player, config)\n        if next_move == None:\n            reward = 0.0\n        else:\n            reward = discount_factor * simulate(next_move, next_player, next_grid, obs, config)\n    \n    update_estimate(grid_to_board(next_grid), reward)\n    return reward\n    \n    \nepisodes = []\n    \ndef agent_monte_carlo(obs, config):\n    \n    deadline = time.time() + config.actTimeout - 0.5\n#    deadline = time.time() + 1\n    \n    grid = board_to_grid(obs.board, config)\n    \n    valid_moves = [col for col in range(config.columns) if grid[0][col] == 0]\n    \n    k = 0\n    while time.time() < deadline:\n        move = random.choice(valid_moves)        \n        simulate(move, obs.mark, grid, obs, config)\n        episodes.append(k)\n        k+=1\n\n    best_val = -1\n    best_move = 0\n    for move in valid_moves:\n        val = get_value(grid_to_board(drop_piece(grid, move, obs.mark, config)))\n        if val >= best_val:\n            best_val = val\n            best_move = move\n    \n    return best_move')


# In[ ]:


# # Two agents play one game round
# env.run([agent_monte_carlo, agent_monte_carlo]);
# # Show the game
# env.render(mode="ipython")


# In[ ]:


# env.play([None, agent_monte_carlo])


# In[ ]:


get_ipython().run_line_magic('run', 'submission.py')


# In[ ]:


# Validate submission file

import sys
from kaggle_environments import utils

out = sys.stdout
submission = utils.read_file("/kaggle/working/submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, "random"])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")


# In[ ]:



