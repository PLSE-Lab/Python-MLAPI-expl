#!/usr/bin/env python
# coding: utf-8

# # Is this Notebook Running Interactively?
# 
# I often use "toy mode" toggles to limit how much data I work with. I never want to commit kernels in toy mode, so I was wondering if I could check the current environment programmatically. Here are some solutions!
# 
# Discussion: https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88158

# ## Solution 1: IPython connection file
# The `connection_file` looks like '/tmp/.local/share/jupyter/runtime/kernel-b2d3dd07-bd5b-4abc-9b8b-2c73511d7281.json' in notebooks and like '/tmp/tmpy7lhke4y.json' when commited.

# In[1]:


def is_interactive():
   return 'runtime' in get_ipython().config.IPKernelApp.connection_file

print('Interactive?', is_interactive())


# ## Solution 2: Environment variable
# 
# Found by Benjamin Minixhofer. 3 environment variables are set when submitting but not when in interactive mode. These are:
# 
# `['PWD', 'SHLVL', '_']`

# In[3]:


import os
def is_interactive():
   return 'SHLVL' not in os.environ

print('Interactive?', is_interactive())


# **Keep in mind there's no guarantee these will keep working!**
