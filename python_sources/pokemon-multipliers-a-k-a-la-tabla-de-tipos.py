#!/usr/bin/env python
# coding: utf-8

# # Pokemon Multipliers
# 
# Este Kernel esta destinado a calcular los multiplicadores de la **tabla de tipos** a las batallas pokemons.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import re
to_underscore = lambda x: re.sub("[^0-9a-zA-Z#]+", "_", x.lower())


# In[ ]:


typetable = pd.read_csv("../input/pokemon-type-table/typetable.csv")
vals = []

for c1 in typetable.columns[1:]:
    # valores para cuando el pokemon defensor solo tiene un tipo
    vals.append(pd.DataFrame({
        "idx": typetable["atck"].map(lambda x: "%s-vs-%s-None" % (x, c1)), #.rename(None)
        "mul": typetable[c1],
    }))
    # valores para cuando el pokemon defensor tiene doble tipo
    for c2 in typetable.columns[1:]:
        vals.append(pd.DataFrame({
            "idx": typetable["atck"].map(lambda x: "%s-vs-%s-%s" % (x, c1, c2)), #.rename(None)
            "mul": typetable[c1] * typetable[c2],
        }))

# pasamos el dataset a un diccionario para que sea mas rapido acceder
mult = pd.concat(vals).reset_index().drop(["index"], axis=1)
mult = dict(zip(mult.values[:,0], mult.values[:,1]))
def multiplier(cat):
    # podemos tanto devolver nan, como devolver directamente 0 cuando no tenemos valor
    return mult.get(cat, 0)
    # return mult.get(cat, np.nan)


# In[ ]:


print(multiplier("Water-vs-Fire-None"))
print(multiplier("Water-vs-Fire-Grass"))
print(multiplier("Fire-vs-Water-Fire"))
print(multiplier("Fire-vs-Grass-Bug"))
print(multiplier("None-vs-Grass-Bug"))


# In[ ]:


pokemon = pd.read_csv("../input/pokemon-challenge-mlh/pokemon.csv").rename(to_underscore, axis='columns').fillna("None")
pokemon["legendary"] = pokemon["legendary"].map(int)
pokemon = pokemon.drop(["hp", "attack", "defense", "sp_atk", "sp_def", "speed", "generation", "legendary"], axis=1)
pokemon.head()


# In[ ]:


battles = pd.read_csv("../input/pokemon-challenge-mlh/battles.csv").rename(to_underscore, axis='columns')
battles.head()


# In[ ]:


def merge_data(battles):
    # hacemos el merge
    battles = battles         .merge(pokemon.rename(lambda x: "f_%s" % x, axis="columns"), left_on="first_pokemon", right_on="f_#")         .merge(pokemon.rename(lambda x: "s_%s" % x, axis="columns"), left_on="second_pokemon", right_on="s_#") 
    # aplicamos los multiplicadores
    battles["f_t1"] = (battles["f_type_1"] + "-vs-" + battles["s_type_1"] + "-" + battles["s_type_2"]).map(multiplier)
    battles["f_t2"] = (battles["f_type_2"] + "-vs-" + battles["s_type_1"] + "-" + battles["s_type_2"]).map(multiplier)
    battles["s_t1"] = (battles["s_type_1"] + "-vs-" + battles["f_type_1"] + "-" + battles["f_type_2"]).map(multiplier)
    battles["s_t2"] = (battles["s_type_2"] + "-vs-" + battles["f_type_1"] + "-" + battles["f_type_2"]).map(multiplier)
    
    # eliminamos los datos originales
    battles = battles        .sort_values(['battle_number'])         .reset_index()         .drop(["index","battle_number", "first_pokemon", "second_pokemon", "f_#", "s_#"], axis=1)
    return battles

train = merge_data(battles)

train.head(50)


# In[ ]:





# In[ ]:




