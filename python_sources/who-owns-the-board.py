#!/usr/bin/env python
# coding: utf-8

# This notebook uses a file from [the dataset](https://www.kaggle.com/jpmiller/connect-four-datasets) I posted. It shows one way to transform the opening moves file into something easily used for machine learning. At this time, I don't beleive you can use the file within the actual environment for competition.
# 
# The file description has more info on the file. Here's the gist:
# 
#   - Columns of the board are labeled left to right, a through g.
#   - Rows of the board are numbered 1 to 6 from bottom to top. 
#   
#   - Columns in the data match the 42 positions above with the order of a1, a2 .. g5, g6.
#   - The last column represents the theoretical value of who should win the game.
#   
# The new structure will be a 3-d numpy array of stacked game boards. Axis 0 is along the stack of boards, axis 1 is along the rows of a single board, axis 2 is across the columns on the board. Final dimensions will be n games * 6 rows * 7 columns.

# In[ ]:


import numpy as np
np.set_printoptions(edgeitems=200)
import pandas as pd
pd.set_option("display.max_columns", 200)
import string


# In[ ]:


# Make column names so that board rows are numbered top-down
numbers = np.arange(6,0,-1)
letters = list(string.ascii_lowercase[0:7])
colnames = [l+str(n) for l in letters for n in numbers]

df = pd.read_csv("../input/connect-four-datasets/connect4_openings.csv", 
                 header=None, names=colnames+['winner']).sort_index(axis=1)

outcomes = df.pop('winner')
df.head()


# In[ ]:


# Change values to integer
player_dict = {'x': 1,
               'o': -1,
               'b': 0
               }

df = df.replace(player_dict)


# In[ ]:


# Reshape
new_board = np.reshape(df.to_numpy(), (-1, 7,6))
new_board = np.swapaxes(new_board, 1, 2)
new_board[0:2]


# Now we can feed boards in batches to pytorch or whatever.
