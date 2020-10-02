#!/usr/bin/env python
# coding: utf-8

# # Install kaggle-environments

# In[ ]:


# 1. Enable Internet in the Kernel (Settings side pane)

# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 
# !curl -X PURGE https://pypi.org/simple/kaggle-environments

# ConnectX environment was defined in v0.1.6
get_ipython().system("pip install 'kaggle-environments>=0.1.6'")


# # Create an Agent
# 
# To create the submission, an agent function should be fully encapsulated (no external dependencies).  
# 
# When your agent is being evaluated against others, it will not have access to the Kaggle docker image.  Only the following can be imported: Python Standard Library Modules, gym, numpy, scipy, pytorch (1.3.1, cpu only), and more may be added later.
# 
# 

# In[ ]:


def my_agent(observation, configuration):
    import numpy as np
    import re
    from scipy.linalg import hankel, toeplitz
    
    def get_opponent(player):
        assert player in [1, 2], "valid player values: {1, 2}"

        # transform to range {0, 1}
        player -= 1

        # get opponent by adding 1 mod 2
        opponent = (player + 1) % 2

        # transform back to range {1, 2}
        opponent += 1

        return opponent


    def array2string(arr):
        arr = np.array(arr)
        return "".join(list(arr.astype(int).astype(str)))

    
    def rows_to_board_indices(matches):
        '''
        return the board coordinates for a list of tuples containing list_indices and match_indices:

        - matches (list): [(list_index, match_index), (list_index, match_index) .. ]
        '''

        output = []
        for list_index, match_index in matches:
            output.append((list_index, match_index))
        return output


    def cols_to_board_indices(matches):
        '''
        return the board coordinates for a list of tuples containing list_indices and match_indices:

        - matches (list): [(list_index, match_index), (list_index, match_index) .. ]
        '''

        output = []
        for list_index, match_index in matches:
            output.append((match_index, list_index))
        return output


    def diag_to_board_indices(board, matches, direction):
        '''
        return board indices for diagonal hits.

        - board (ndarray): numpy 2D array representing board state
        - matches (list): [(list_index, match_index), (list_index, match_index) .. ]
        - direction (str):
            - 'bltr': bottom-left top-right diagonals
            - 'tlbr': top-left bottom-right diagonals
        ''' 
        assert direction in ["bltr", 'tlbr']

        # get dimensions
        nrows, ncols = board.shape

        # initialize first column and last row to set up diagonal matrix
        first_col = np.arange(nrows)
        last_row = np.arange(nrows-1, nrows + ncols-1)

        if direction == "bltr":
            # use toeplitz matrix for the bottom-left to top-right diagonal
            diagonal_matrix = toeplitz(first_col[::-1], last_row)
        else:
            # use hankel matrix for the top-left to bottom-right diagonals
            diagonal_matrix = hankel(first_col, last_row)

        output = []
        for list_index, match_index in matches:
            diagonal_board_indices = np.where(diagonal_matrix == list_index)
            r = diagonal_board_indices[0][match_index]
            c = diagonal_board_indices[1][match_index]
            output.append((r, c))
        return output
    

    def find_pattern(list_of_lists, pattern):
        '''
        Find pattern (str) in a list of arrays.

        Returns list of lists with start indices of each match.

        Examples:
            >>> matches = find_pattern([[1, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1, 1]], "11")
            >>> for match in matches:
            >>>    print(match)
            [3]
            []
            [0, 2]
            [2, 5]
        '''

        all_matches = []
        for i, arr in enumerate(list_of_lists):
            string = array2string(arr)
            all_matches.append([(m.start()) for m in re.finditer(pattern, string)])
        return all_matches


    def find_zeros_pattern(list_of_lists, pattern):
        '''
        Find pattern (str) in a list of arrays that are actionable. This function assumes that the pattern contains zeros ("0")
        This function finds the pattern in the list of lists, and checks if the zeros that are part of the match are actionable, which means
        that there exists a valid action, to fill the corresponding cell.

        Returns list of lists with start indices of each match.

        Examples:
            >>> matches = find_actionable_pattern([[1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 0, 1, 0]], "1010")
            >>> print(matches)
            [(0, 1), (0, 5), (2, 4), (0, 3), (0, 7), (2, 6)]

        In each tuple the first value is the list index, the second value is the index of a zero-value in that list.
        '''

        # find all matches
        matches = find_pattern(list_of_lists, pattern)

        # get indices of the zeros in the pattern
        zeros_in_pattern = [z.start() for z in re.finditer("0", pattern)]

        results = []
        for zero in zeros_in_pattern:
            for i, match in enumerate(matches):
                if len(match) > 0:
                    for entry in list(np.array(match) + zero):
                        results.append((i, entry))
        return results


    def is_actionable(board, r, c):
        '''
        return true if the (r, c) cell can be reached by making an action (a=c)
        '''
        if (board[r, c] == 0) and (0 not in board[r+1:, c]):
            return True
        return False


    def filter_actionable(board, coords):
        actionable_coords = []
        for r, c in coords:
            if is_actionable(board, r, c):
                actionable_coords.append((r, c))
        return actionable_coords


    def get_board_lines(board):
        nrows = board.shape[0]
        ncols = board.shape[1]        
    
        rows = [board[i, :] for i in np.arange(nrows)]
        cols = [board[:, j] for j in np.arange(ncols)]

        # offset ranges to get all diagonals for both bottom-left-top-right (bltr) and top-left-bottom-right (tlbr)
        offset_bltr = {
            'start': -(nrows-1), 
            'stop': ncols
        } 
        offset_tlbr = {
            'start': -(ncols-1), 
            'stop': nrows
        }
        
        diag1 = [np.diagonal(board, offset=offset) for offset in np.arange(**offset_bltr)]
        diag2 = [np.diagonal(np.rot90(board), offset=offset) for offset in np.arange(**offset_tlbr)]    
    
        return {'rows': rows, 'cols': cols, 'diag1': diag1, 'diag2': diag2}

    
    def find_actionable_patterns(board, pattern):
        
        # get rows, columns and diagonal lines
        all_lists = get_board_lines(board)

        results = []
        actionable_results = []    

        for list_type, list_of_lists in all_lists.items():
            zero_indices = find_zeros_pattern(list_of_lists, pattern)
            if list_type == 'rows':
                board_coords = rows_to_board_indices(zero_indices)
            if list_type == 'cols':
                board_coords = cols_to_board_indices(zero_indices)
            if list_type == 'diag1':
                board_coords = diag_to_board_indices(board, zero_indices, direction="bltr")
            if list_type == 'diag2':
                board_coords = diag_to_board_indices(board, zero_indices, direction="tlbr")
            results.extend(board_coords)
            actionable_results.extend(filter_actionable(board, board_coords))
        return results, actionable_results
    
    ##############
    ### look ahead
    
    def get_row_of_first_piece(column):
        # get row of first piece
        pieces_in_column = np.nonzero(column)
        if np.size(pieces_in_column) > 0:
            return np.min(pieces_in_column)
        else:
            return len(column)


    def get_next_state(board, player, action):
        assert action in np.arange(board.shape[1]), f"invalid action {action}. Valid range: {0}-{board.shape[1]-1}"

        # copy board, to preserve original state
        board = board.copy()

        # column where action is taken 
        column = board[:, action]

        # get row for first piece in the column (top-down)
        row_first_piece = get_row_of_first_piece(board[:, action])

        # test for full columns
        assert row_first_piece >= 0, f"invalid action {action}, column is full, row_first_piece: {row_first_piece}"

        # add piece for player to cell above
        board[row_first_piece-1, action] = player

        return board

    
    def cartesian_product(num_actions, num_levels):
        '''
        Return tree with all possible actions and counter actions.

        Example:

        >>> cartesian_product(num_actions=3, num_levels=2)
        >>> array([[0, 0],
           [0, 1],
           [0, 2],
           [1, 0],
           [1, 1],
           [1, 2],
           [2, 0],
           [2, 1],
           [2, 2]])
        '''
        arrays = np.repeat([np.arange(num_actions)], num_levels, axis=0)

        num_arrays = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [num_arrays], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, num_arrays)
    
    
    def game_finished(board, inarow=4):
        all_lines = get_board_lines(board)

        # look for winning pattern accross all lines
        for line_type, line in all_lines.items():
            for player in ['1', '2']:
                found = find_pattern(line, inarow * player)
                if len(np.nonzero(found)[0]):
                    return True, int(player)
        return False, None

    
    class Observation():
        board = None
        mark = 1


    class Configuration():
        columns = 7
        rows = 6
        inarow = 4


    def get_losing_moves(board, player):
        nrows, ncols = board.shape
        num_actions = ncols
        num_levels = 2
        opponent = get_opponent(player)
        sim_observation = Observation()
        sim_observation.board = board
        sim_observation.mark = player

        sim_configuration = Configuration()

        # get cartesian product, and reshape it group it by player action
        action_tree = cartesian_product(num_actions, num_levels)

        losing_moves = []

        for trajectory in action_tree:
            # reset observation to current state and make sure player is next to move
            sim_observation.board = list(np.ravel(board))
            sim_observation.mark = player

            for action in trajectory:
                if sim_observation.board[action] != 0:
                    # invalid action
                    continue

                # simulate a move
                next_state = get_next_state(np.array(sim_observation.board).reshape(nrows, ncols), sim_observation.mark, action)

                # update board
                sim_observation.board = list(np.ravel(next_state))

                # switch mark to opponent
                sim_observation.mark = get_opponent(sim_observation.mark)

                done, won = game_finished(next_state, inarow=4)
 
                if done == True and won == opponent:
                    losing_moves.append(trajectory[0])
                    break
        return losing_moves
    
    ##################
    ### end look ahead
    
    
    def return_action():
        # shape board
        nrows = configuration['rows']
        ncols = configuration['columns'] 
        
        board = np.array(observation['board']).reshape((nrows, ncols))
        player = int(observation['mark'])
        opponent = get_opponent(player)
        
        # define patterns to search for
        patterns = [
            # winning / losing moves
            {'string': '1110', 'weight': 1},
            {'string': '1101', 'weight': 1},
            {'string': '1011', 'weight': 1},             
            {'string': '0111', 'weight': 1},
            
            {'string': '110', 'weight': 0.4},
            {'string': '101', 'weight': 0.4},
            {'string': '011', 'weight': 0.4},            
        ]
        
        progressing_moves = []
        preventing_moves = []
        
        for pattern in patterns:            
            # find the patterns accross columns, rows, diagonals
            results_player, actionable_player_moves = find_actionable_patterns(board, pattern['string'].replace('1', str(player)))
            results_player, actionable_opponent_moves = find_actionable_patterns(board, pattern['string'].replace('1', str(opponent)))
            
            for move in actionable_player_moves:
                progressing_moves.append({'action': move[1], 'weight': pattern['weight'], 'is_player': 1, 'string': pattern['string']})
            
            for move in actionable_opponent_moves:
                preventing_moves.append({'action': move[1], 'weight': pattern['weight'], 'is_player': 0, 'string': pattern['string']})
           
        all_moves = progressing_moves + preventing_moves
        
        losing_moves = get_losing_moves(board, player=player)
        
        sorted_moves = sorted(all_moves, key=lambda k: (k['weight'], k['is_player']), reverse=True)
        # print("1. sorted_moves:", sorted_moves)
        sorted_moves = [move for move in sorted_moves if not move['action'] in losing_moves]
        # print("2. sorted_moves:", sorted_moves)
        
        # print("losing_moves:", losing_moves, "player",player)
        
        if len(sorted_moves) > 0:
            action = int(sorted_moves[0]['action'])
            # print("sorted move:", action)
        else:
            if np.sum(board) == 0:
                # return start move
                return 3
            else:
                ######################################################
                # choose random action
                from random import choice
                moves = [c for c in range(configuration.columns) if observation.board[c] == 0 and c not in losing_moves]
                if len(moves) > 0:
                    action = choice(moves)
                else:
                    # all lost..
                    action = choice([c for c in range(configuration.columns) if observation.board[c] == 0])
        
        # print("action:", action)
        return action
    return return_action()


# In[ ]:


from kaggle_environments import evaluate, make, utils

e = make("connectx", debug=True)
e.reset()

# Play as first position againt my_agent (self play)
t = e.train([my_agent, None])

o = t.reset()

print("start state")
e.render()

while not e.done:
    a = my_agent(o, e.configuration)
    #print("action", a)
    o, r, d, i = t.step(a)

print("end state, reward")    
e.render()

print("* Conclusion: player 1 wants to extend its own 2-cell line on (5, 2), (5,3)) helping player 2 to make a winning move.")
print("* Solution would be to do a 1-step look ahead, and check if an action leads to winning moves for the opponent.")


# In[ ]:


# from kaggle_environments import evaluate, make, utils

# env = make("connectx", debug=True)
# env.reset()

# # Play as first position againt my_agent (self play)
# trainer = env.train([None, my_agent])

# observation = trainer.reset()
# #observation.board = list(board.ravel().astype(int))

# print("start state")
# env.render()

# while not env.done:
#     my_action = my_agent(observation, env.configuration)
#     print("action", my_action)
#     observation, reward, done, info = trainer.step(my_action)

# print("end state, reward")    
# env.render()

# print("* Conclusion: player 1 wants to extend its own 2-cell line on (5, 2), (5,3)) helping player 2 to make a winning move.")
# print("* Solution would be to do a 1-step look ahead, and check if an action leads to winning moves for the opponent.")


# # Test your Agent

# In[ ]:


from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)
env.reset()
# Play as the first agent against default "random" agent.
#env.run([my_agent, "random"])
env.run([my_agent, "negamax"])
env.render(mode="ipython", width=500, height=450)


# # Evaluate your Agent

# In[ ]:


def mean_reward(rewards):
    #print(rewards)
    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)

# # Run multiple episodes to estimate it's performance.
print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, my_agent], num_episodes=20)))
# print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))
# # print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent_rule02, "negamax"], num_episodes=10)))
# # print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent_rule02, "negamax"], num_episodes=10)))


# # Write Submission File

# In[ ]:


import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(my_agent, "submission.py")


# # Submit to Competition
# 
# 1. Commit this kernel.
# 2. View the commited version.
# 3. Go to "Data" section and find submission.py file.
# 4. Click "Submit to Competition"
# 5. Go to [My Submissions](https://kaggle.com/c/connectx/submissions) to view your score and episodes being played.

# ### Validate submission file

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

