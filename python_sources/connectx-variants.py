#!/usr/bin/env python
# coding: utf-8

# # Configuring ConnectX Board for Custom Dimensions

# Many people in the discussion boards have discussed an interest in competing on alternative boards.  In advance of a potential switch to other board sizes, here are some examples for how to set up the environment with the `0.1.6` API with custom board sizes.

# # Installing the latest environment

# In[ ]:


# We must use an internet-connected kernel to download the latest version of the environment
get_ipython().system('pip install "kaggle_environments==0.1.6"')


# In[ ]:


import kaggle_environments as ke


# # Making new environments

# In the `kaggle_environments` API, we create new instances of the `Environment` class using the `make` command in: https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/core.py.  The `make` function appears to be a wrapper around the `Environment` initialization function, validating inputs.  The command takes four parameters:
# * `environment` - this can be a string for one of the folders in in `kaggle_environments.envs` (https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs) or an instance of an Environment.  Currently, there are only three types: ConnectX, TicTacToe and an Identity environment - this last one appears to be a dummy for testing.
# * `configuration` \[optional\] - this is a `Dict` specifying game parameters used by the game engine specified above.  I will spend more on this time later.
# * `steps` \[optional\] - this is a `list` of game states to fastforward the game board, if empty, the initial state is used by default.  Potentially, this could be used to implement the ["5-in-a-row"](https://en.wikipedia.org/wiki/Connect_Four#5-in-a-Row) variant.
# * `debug` \[optional\] - this is a flag that determines whether the engine will print various messages to console during the call to `step()`; used by the private function `__debug_print`

# # Using a custom configuration

# The `configuration` dictionary provided to `make` is where we can specify new board parameters. The default configuration can be found within the specification JSON file here:[kaggle_environments/envs/connectx/connectx.json](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/connectx/connectx.json).  I've copied the entire `0.1.6` version of the file below for convenience:

# ```json
# {
#   "name": "connectx",
#   "title": "ConnectX",
#   "description": "Classic Connect in a row but configurable.",
#   "version": "1.0.0",
#   "agents": [2],
#   "configuration": {
#     "timeout": {
#       "description": "Seconds an agent can run before timing out.",
#       "type": "integer",
#       "minimum": 1,
#       "default": 5
#     },
#     "columns": {
#       "description": "The number of columns on the board",
#       "type": "integer",
#       "default": 7,
#       "minimum": 1
#     },
#     "rows": {
#       "description": "The number of rows on the board",
#       "type": "integer",
#       "default": 6,
#       "minimum": 1
#     },
#     "inarow": {
#       "description": "The number of checkers in a row required to win.",
#       "type": "integer",
#       "default": 4,
#       "minimum": 1
#     }
#   },
#   "reward": {
#     "description": "0 = Lost, 0.5 = Draw, 1 = Won",
#     "enum": [0, 0.5, 1],
#     "default": 0.5
#   },
#   "observation": {
#     "board": {
#       "description": "Serialized grid (rows x columns). 0 = Empty, 1 = P1, 2 = P2",
#       "type": "array",
#       "items": {
#         "enum": [0, 1, 2]
#       },
#       "default": []
#     },
#     "mark": {
#       "default": 0,
#       "description": "Which checkers are the agents.",
#       "enum": [1, 2]
#     }
#   },
#   "action": {
#     "description": "Column to drop a checker onto the board.",
#     "type": "integer",
#     "minimum": 0,
#     "default": 0
#   },
#   "reset": {
#     "status": ["ACTIVE", "INACTIVE"],
#     "observation": [{ "mark": 1 }, { "mark": 2 }]
#   }
# }
# ```

# We primarily care about the `configuration` section of the specification.  The elements that we can modify are:
# * `timeout` - controls the maximum time of each agent, 5 is the default
# * `columns` - controls how wide the board is, 7 is the default
# * `rows` - controls how tall the board is, 6 is the default
# * `inarow` - controls how many markers must be connected to win, 4 is the default

# In order to customize our boards, we need to pass a custom configuration dictionary to `make`.  We do not need to specify all of the configuration parameters, only those whose default values we wish to override.  As an example:

# In[ ]:


config_9x7 = {
    'columns': 9,
    'rows': 8,
    'inarow': 5
}
env_9x7 = ke.make('connectx', configuration=config_9x7)
env_9x7.render()


# In[ ]:


config_5x4 = {
    'columns': 5,
    'rows': 4,
    'inarow': 3
}
env_5x4 = ke.make('connectx', configuration=config_5x4)
env_5x4.render()


# In[ ]:


config_8x8 = {
    'columns': 8,
    'rows': 8,
    'inarow': 4
}
env_8x8 = ke.make('connectx', configuration=config_8x8)
env_8x8.render()


# # End

# Using these new environment variations, we can train our agents on custom board sizes to develop configuration-agnostic algorithms in anticipation of future changes to the default board size.
