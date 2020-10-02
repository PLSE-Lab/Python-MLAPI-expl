#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from learntools.core import binder
binder.bind(globals())
from learntools.game_ai.ex1 import


# In[ ]:


def drop_piece(grid, col, piece, config):
    next_grid = grid.copy()
    for row in range(config.rows-1, -1, -1):
        if next_grid[row][col] == 0:
            break
    
    next_grid[row][col] = piece
    return next_grid


# In[ ]:


def wining_move(obs, config, col, piece):
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    next_grid = drop_piece(grid, col, piece, config)
    
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(next_grid[row, col:col+config.inarow])
            if window.count(piece) == config.inarow:


# In[ ]:





# In[ ]:





# In[ ]:




