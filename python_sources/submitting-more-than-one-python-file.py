#!/usr/bin/env python
# coding: utf-8

# # Submitting more than one Python file

# ## Situation
# 
# Let's assume you have a project structure that looks like this:
# 
# ```shell
# connect_x/
#     __init__.py
#     game.py
#     agent.py
#     utils.py
# submission.py
# ```
# 
# Where your `submission.py` might look a little something like this:
# 
# ```python
# from connect_x import game, agent, utils
# 
# 
# def act(observation, configuration):
#     pass
# ```

# ## Problem
# 
# Kaggle only allows **single file submissions**. This means that you cannot submit a Python script that depends on other modules.
# 
# **Seems like we will have to manually copy and paste our code into one `submission.py`, right?**

# ## Building one submission from multiple files
# 
# Wrong! We can use the [stickytape](https://github.com/mwilliamson/stickytape) package here.
# 
# > Stickytape can be used to convert a Python script and any Python modules it depends into a single-file Python script.

# ## Usage
# 
# ```shell
# pip install stickytape
# 
# stickytape submission.py > submission_standalone.py
# ```
# 
# The `submission_standalone.py` now contains the source code from the imported modules. It can be submitted to this competition:
# 
# ```shell
# kaggle competitions submit \
#     -f submission_standalone.py \
#     connectx
# ```
