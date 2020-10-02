#!/usr/bin/env python
# coding: utf-8

# # Analysing Halite Matches
# # _Part 1: Getting the data right with pandas_

# ## Motivation

# Many top tier players of the Halite 3 competition have stated that analysing the behaviour of their bots was a key ingredient in their success. For example, this is the "battlestation" of #3 player [reCurse](https://recursive.cc/blog/halite-iii-postmortem.html):
# 
# ![reCurse's battlestation](https://recursive.cc/blog/halite-iii-battlestation.png)
# 
# 
# 
# With this notebook, I am attemtping to start a series that will demonstrate how to get the most information out of the environment data. I am trying to write three notebooks:
# 
# 1. How to extract and transform the data from the environment (this one!)
# 1. How to store the extracted data in Google Big Query and build dashboards with DataStudio (stay tuned!)
# 1. How to build dashboards with Jupyter and Voila (stay tuned!)
# 
# Maybe at the end of this project we will have our own "battlestation" or something similar. :)

# **In this notebook you will learn:**
# 
# - How to convert all environment information to pandas dataframes.
# - How to add "MINE" and "DEPOSIT" actions to the ships.
# - How to extract collisions for the environment.
# - And much more!
# 

# ## General Information
# 
# ### Pandas pipelines
# 
# In this notebook, I use a lot of pipelining in Pandas. If you have not heard of Pandas pipelines before, please refer to this fantastic [talk](https://www.youtube.com/watch?v=yXGCKqo5cEY) from [Vicent Warmerdam](https://twitter.com/fishnets88).
# 
# ### UID bug
# 
# Currently there is [a bug](https://github.com/Kaggle/kaggle-environments/issues/15) in the interpreter, which stops the step count from incrementing in the first turn. This causes the `uid` to be non-unique, when units are created in the second turn. This also affects this analysis. Don't be too surprised when seeing duplicate uid. I won't attempt to patch this here because I am assuming this will be resolved in the next release.
# 
# ### WIP
# 
# This is a work in progress. Right now, there is not a lot of explanatory comments yet. I will try to add these over the next days. All plots only serve to show to potential of this data. Proper plotting will be done in the next notebooks.
# 
# **Feedback is very appreciated!**

# ## Setup

# In[ ]:


get_ipython().system(' pip install kaggle_environments==0.2.1 mypy')


# In[ ]:


from typing import List
import json


# In[ ]:


from kaggle_environments import make
from kaggle_environments.envs.halite.halite import get_to_pos
from kaggle_environments.utils import Struct
import pandas as pd


# ### Building the example match

# I was having difficulties presenting a reproducible match, even when setting the random number seed. So I decided to "pre-play" a match and store the environment steps here. The game was configured as such:
# 
# ```python
# env = make("halite")
# _ = env.reset(num_agents=4)
# _ = env.run(["random", "random", "random", "random"])
# ```

# In[ ]:


env = make("halite")
_ = env.reset(num_agents=4)


# In[ ]:


def convert_to_struct(obj):
    """
    Converts an object to the Kaggle `Struct` format.
    """
    if isinstance(obj, list):
        return [convert_to_struct(item) for item in obj]
    if isinstance(obj, dict):
        return Struct(**{key: convert_to_struct(value) for key, value in obj.items()})
    return obj


with open("../input/halite-match-steps/steps.json", mode="r") as file_pointer:
    env.steps = convert_to_struct(json.load(file_pointer))


# Take a look at the replay. You should compare it against the insights the we find throughout this notebook sometimes, it really helps to make sure that everything is working well.

# In[ ]:


env.render(mode="ipython", width=800, height=600)


# ## Bringing the data into the right format

# At first, let's build some quite straightforward tables for players, actions ships and shipyards. They will serve as a foundation for our more complicated tables.

# ### Basic tables for players, actions, ships and shipyards

# In[ ]:


def make_actions_df(steps: List[List[Struct]]) -> pd.DataFrame:
    
    def actions_from_steps(steps):
        for step, state in enumerate(steps):
            step = state[0].observation.step
            for player_index, player in enumerate(state):
                for uid, action in player.action.items():
                    yield {"step": step, "uid": uid, "action": action}
                    
    return pd.DataFrame(actions_from_steps(steps))


def make_ships_df(steps: List[List[Struct]]) -> pd.DataFrame:
    
    def ships_from_steps(steps):
        for step, state in enumerate(steps):
            step = state[0].observation.step
            for player_index, player in enumerate(state[0].observation.players):
                for uid, (pos, halite) in player[2].items():
                    yield {"step": step, "uid": uid, "pos": pos, "halite": halite, "player": player_index}
                    
    return pd.DataFrame(ships_from_steps(steps))

def make_shipyards_df(steps: List[List[Struct]]) -> pd.DataFrame:
    
    def shipyard_from_state(state):
        for step, state in enumerate(steps):
            step = state[0].observation.step
            for player_index, player in enumerate(state[0].observation.players):
                for uid, pos in player[1].items():
                    yield {"step": step, "uid": uid, "pos": pos, "player": player_index}
                
    return pd.DataFrame(shipyard_from_state(steps[-1]))

def make_players_df(steps: List[List[Struct]]) -> pd.DataFrame:
    
    def players_from_steps(steps):
        for step, state in enumerate(steps):
            step
            for player_index, player in enumerate(state[0].observation.players):
                yield {"step": step, "player": player_index, "halite": player[0]}
                
    return pd.DataFrame(players_from_steps(steps))


# In[ ]:


actions_df = make_actions_df(env.steps)
actions_df.head()


# In[ ]:


ships_df = make_ships_df(env.steps)
ships_df.head()


# In[ ]:


shipyards_df = make_shipyards_df(env.steps)
shipyards_df.head()


# In[ ]:


players_df = make_players_df(env.steps)
players_df.head()


# ### Advanced tables

# In[ ]:


# Some pipeline functions for our advanced tables.

def add_halite_delta(df: pd.DataFrame) -> pd.DataFrame:
    def _halite_delta(ship):
        ship = ship.sort_values("step", ascending=True)
        return ship["halite"] - ship.shift()["halite"]
    df["halite_delta"] = df.groupby("uid").apply(_halite_delta).reset_index("uid")["halite"]
    return df
    
def add_mine_deposit_actions(df: pd.DataFrame) -> pd.DataFrame:
    shipyard_present = ~pd.isna(
        df
        .merge(shipyards_df, how="left", on=["step", "pos"], suffixes=["_ship", "_shipyard"])
        ["uid_shipyard"]
    )
    
    filter_ = (pd.isna(df["action"])) & (~pd.isna(df["halite_delta"]))
    
    df.loc[filter_ & shipyard_present, "action"] = "DEPOSIT"
    df.loc[filter_ & (~shipyard_present), "action"] = "MINE"

    return df

def add_halite_delta_abs(df: pd.DataFrame) -> pd.DataFrame:
    df["halite_delta_abs"] = df["halite_delta"].abs()
    return df

def add_step_prev(df: pd.DataFrame) -> pd.DataFrame:
    df["step_prev"] = df["step"] - 1
    return df

def add_expected_pos(df: pd.DataFrame) -> pd.DataFrame:
    df["expected_pos"] = df.apply(lambda ship: get_to_pos(env.configuration.size, ship["pos_prev"], ship["action"]), axis=1)
    return df


# #### Ship action table

# Mining and depositing halite is done implicitly by not adding an action to a ship. For us this means that we need to figure out whether a ship was mining or depositing when no action was provided. My approach is checking whether we are sharing the cell with a shipyard.

# In[ ]:


ship_actions_df = (
    actions_df
    .copy()
    .pipe(lambda df: df[df["action"].isin(("NORTH", "SOUTH", "EAST", "WEST", "CONVERT"))])
    .merge(ships_df, how="outer", on=["step", "uid"])
    .pipe(add_halite_delta)
    .pipe(add_mine_deposit_actions)
)


# In[ ]:


ship_actions_df.head()


# In[ ]:


# Number of actions sent per action type.
(
    ship_actions_df
    .groupby("action")
    .size()
    .sort_values(ascending=False)
    .plot(kind="bar")
)


# In[ ]:


# Amount of halite minded (deducting the halite spent on moving) per ship.
(
    ship_actions_df
    [ship_actions_df["action"] != "DEPOSIT"]
    .groupby("uid")
    ["halite_delta"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .plot(kind="bar")
)


# #### Shipyard actions

# In[ ]:


shipyard_actions_df = (
    actions_df
    .copy()
    .pipe(lambda df: df[df["action"].isin(("SPAWN", ))])
    .merge(shipyards_df, how="right", on=["step", "uid"])
)


# In[ ]:


shipyard_actions_df.head()


# In[ ]:


# Number of spawn actions by shipyard.
(
    shipyard_actions_df
    [shipyard_actions_df["action"] == "SPAWN"]
    .groupby("uid")
    .size()
    .sort_values(ascending=False)
    .head()
    .plot(kind="bar")
)


# #### Deposit table

# Next I would like to see which shipyards receive the most halite deposists.

# In[ ]:


deposit_df = (
    ship_actions_df
    [(ship_actions_df["action"] == "DEPOSIT") & (~pd.isna(ship_actions_df["halite_delta"]))]
    .merge(shipyards_df, how="left", on=["step", "pos"], suffixes=["_ship", "_shipyard"])
    .pipe(add_halite_delta_abs)
    [["step", "pos", "uid_ship", "uid_shipyard", "player_ship", "halite_delta_abs"]]
    .rename({"player_ship": "player", "halite_delta_abs": "halite"}, axis=1)
)


# In[ ]:


deposit_df.head()


# In[ ]:


(
    deposit_df
    .groupby("uid_shipyard")
    ["halite"]
    .sum()
    .sort_values(ascending=False)
    .head()
    .plot(kind="bar")
)


# Random agents are not too good at depositing halite ...

# #### Collision table

# A critical piece of information but unfortunately quite difficult to extract: Collisions!

# In[ ]:


ship_collision_df = (
    ship_actions_df
    .groupby("uid")
    .apply(lambda ship: ship.sort_values("step").tail(1))
    .reset_index(drop=True)
    .pipe(add_step_prev)
    .merge(ships_df, how="left", left_on=["uid", "step_prev"], right_on=["uid", "step"], suffixes=["", "_prev"])
    .pipe(add_expected_pos)
    [["step", "uid", "expected_pos"]]
    .rename({"expected_pos": "pos"}, axis=1)
    .append(ships_df[["step", "uid", "pos"]])
    .groupby(["step", "pos"])["uid"].aggregate(lambda x: set(x)).reset_index()
    .pipe(lambda df: df[df["uid"].apply(lambda x: len(x) > 1)])
)

ship_collision_df


# ## Conclusion

# With a little bit of pandas pipelining, we can actually extract a lot of useful information from the environment steps!
