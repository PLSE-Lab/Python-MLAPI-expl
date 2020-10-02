#!/usr/bin/env python
# coding: utf-8

# Each of the plots below has some kind of mistake. See if you can find them!

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast

plt.style.use('seaborn-ticks')


# In[5]:


pokemon_data = pd.read_csv('../input/pokemon-data.csv', 
                           sep = ';', converters={'Types':ast.literal_eval, 'Abilities':ast.literal_eval, 'Moves':ast.literal_eval})
move_data = pd.read_csv('../input/move-data.csv', index_col = 0)
for var in ['Power', 'Accuracy']:
    move_data[var].replace('None', np.nan, inplace=True)
    move_data[var] = move_data[var].astype(float)


# In[6]:


move_data.head()


# In[11]:


for contest in move_data.Contest.unique():
    data_subset = move_data[move_data.Move_Contest == contest]
    plt.scatter(data_subset.Power, 
                data_subset.Accuracy, label = contest)
    plt.xlabel('Power')
    plt.ylabel('Accuracy')
plt.legend(loc = 'lower left', bbox_to_anchor = (1, 0))
plt.show()


# In[13]:


for generation in move_data.Generation.unique():
    print(generation)
    data_subset = move_data[move_data.Generation == generation].dropna()
    subset_label = 'Generation ' + generation
    sns.kdeplot(data_subset.Power, label = subset_label, shade = True)
    plt.xlabel('Power')
    plt.ylabel('How many Pokemon')
plt.show()


# Hint: What does `str` mean in Python?

# In[15]:


plt.scatter(pokemon_data.Attack, 
            pokemon_data['Special Attack'], color = pokemon_data.Defense, cmap = 'cool', alpha = 0.5)
plt.xlabel('Attack')
plt.ylabel('Special Attack')
plt.colorbar(label = 'Defense')
plt.show()


# Hint: Remember that you can see a description of a command using `Shift + Tab`. Click inside the brackets after `plt.scatter`, hit `Shift + Tab` and you'll see a list of arguments that can go into the `scatter` function.

# In[ ]:




