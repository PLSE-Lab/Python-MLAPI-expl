#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system("pip install 'kaggle-environments>=0.1.4'")


# In[ ]:


from kaggle_environments import evaluate, make

env = make("connectx", debug=True)
env.render()


# In[ ]:


# greedy agent
def my_agent(observation, cfg):
    import numpy as np
    
    # https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
    def search_sequence_numpy(arr,seq):
        """ Find sequence in an array using NumPy only.

        Parameters
        ----------    
        arr    : input 1D array
        seq    : input 1D array

        Output
        ------    
        Output : 1D Array of indices in the input array that satisfy the 
        matching of input sequence in the input array.
        In case of no match, an empty list is returned.
        """

        # Store sizes of input array and sequence
        Na, Nseq = arr.size, seq.size

        # Range of sequence
        r_seq = np.arange(Nseq)

        # Create a 2D array of sliding indices across the entire length of input array.
        # Match up with the input sequence & get the matching starting indices.
        M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

        # Get the range of those indices as final output
        if M.any() >0:
            return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
        else:
            return []         # No match found
    
    class Table():
        def __init__(self,rows,cols,inarow,board=None):
            self.inarow = inarow
            if board is None:
                self.board = np.zeros((rows,cols), dtype=int)
            else:
                self.board = np.reshape(board,(rows,cols))
    

        def _winning_rule(self, arr):
            win1rule = np.ones((self.inarow)) # 1 for 1s
            win2rule = np.ones((self.inarow)) + np.ones((self.inarow)) # 2 for 2s

            player1wins = len(search_sequence_numpy(arr,win1rule)) > 0
            player2wins = len(search_sequence_numpy(arr,win2rule)) > 0
            if player1wins or player2wins:
                return True
            else:
                return False

        def _get_diagonals(self, _board, i, j):
            diags = []
            diags.append(np.diagonal(_board, offset=(j - i)))
            diags.append(np.diagonal(np.rot90(_board), offset=-_board.shape[1] + (j + i) + 1))
            return diags

        def _get_axes(self, _board, i, j):
            axes = []
            axes.append(_board[i,:])
            axes.append(_board[:,j])
            return axes

        def _winning_check(self, i, j):
            '''
            Checks if there is four equal numbers in every
            row, column and diagonal of the matrix
            '''    
            all_arr = []
            all_arr.extend(self._get_axes(self.board, i, j))
            all_arr.extend(self._get_diagonals(self.board, i, j))

            for arr in all_arr:
                winner = self._winning_rule(arr)
                if winner:
                    return True
                else:
                    pass
        def check_win(self, player, column, inarow=None):
            if inarow is not None:
                self.inarow=inarow
            colummn_vec = self.board[:,column]
            non_zero = np.where(colummn_vec != 0)[0]

            if non_zero.size == 0:                        
                i = self.board.shape[0]-1
            else:                                          
                i = non_zero[0]-1

            self.board[i,column] = player # make move
            if self._winning_check(i, column):
                return True
            else:
                self.board[i,column] = 0 #unmove
                return False 
            
    us = observation.mark
    them = 2 if us == 1 else 1
        
    table = Table(cfg.rows,cfg.columns,cfg.inarow,board=observation['board'])
    valid_positions = [c for c in range(cfg.columns) if observation.board[c] == 0]

    #optimal first move
    if not np.any(table.board):
        return cfg.columns // 2

    #can we win with a move?
    for move in range(cfg.columns):
        if table.check_win(us,move): 
            #print("win")
            if move in valid_positions: 
                return move
            
    
    #would they with with a move? block it.
    for move in range(cfg.columns):
        if table.check_win(them,move):
            #print("block")
            if move in valid_positions: 
                return move
    
    #check for next greedy move
    for inarow in range(cfg.inarow-1,0,-1):
        for move in range(cfg.columns):
            if table.check_win(us,move,inarow=inarow): 
                if move in valid_positions: 
                    return move 
                
    #first lowest row
    counts = [np.count_nonzero(table.board[:,c]) for c in valid_positions]
    return valid_positions[np.argmin(counts)]


# In[ ]:


def mean_reward(rewards):
    return sum((r[0] or 0.) for r in rewards) / sum((r[0] or 0.) + (r[1] or 0.) for r in rewards)

# Run multiple episodes to estimate it's performance.
for i in range(10):
    print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))


# In[ ]:


import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(my_agent, "submission.py")

