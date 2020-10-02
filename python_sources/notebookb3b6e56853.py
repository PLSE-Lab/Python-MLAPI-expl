#!/usr/bin/env python
# coding: utf-8

# I've always wondered if there was a team that could make an attack that is both super effective AND has the STAB (same type attack bonus) against any Pokemon in the game. This notebook is an exploration to see if that's possible.
# 
# I guess the first thing we should check is whether there are any Pokemon that are not weak to any type:

# In[ ]:


import numpy as np
import pandas as pd

pokemon = pd.read_csv("../input/pokemon.csv")
types = pd.read_csv("../input/type-chart.csv")

type_names = list(types.columns.values)[2:]
query = str.join(" & ", [type_name + " <= 1" for type_name in type_names])
invincible_types = types.query(query)

print("Woohoo!" if len(invincible_types) == 0 else "Aww man!")


# Whew! One boundary case out of the way.
# 
# Now I wonder if there are any Pokemon that are only vulnerable to 1 type...
# 
# To start, let's transform our data into a bit of a friendlier system for analyzing this particular question:

# In[ ]:



vulnerabilities = {
    type_1: {
        type_2: [] for type_2 in types.loc[lambda type: type["defense-type1"] == "normal"]["defense-type2"]
    } for type_1 in type_names
}

print(vulnerabilities)

