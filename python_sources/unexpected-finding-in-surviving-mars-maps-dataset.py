#!/usr/bin/env python
# coding: utf-8

# 
# ## Surviving Mars Maps
# 
# Surviving Mars is a sci-fi city builder all about colonizing Mars and, well, surviving. The Surviving Mars Maps dataset contains a list of all the colony locations with their environmental conditions and breakthroughs available on each map. I've not played the game, nor ever heard of it, but I had some time and wanted to at least take a cursory look at the data. What I found led me to more questions than I had when I started.
# 

# In[ ]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/surviving-mars-maps/MapData-Evans-GP-Flatten.csv', skipinitialspace=True)
totalRows = len(df)
print(str(totalRows) + " rows loaded.")


# In[ ]:


# let's look at the names of all the columns
column_names = df.columns
print(column_names)


# Let's print the unique values appearing in each column.

# In[ ]:


for icol in range(len(column_names)):
    print(column_names[icol])
    print( df[column_names[icol]].unique() )
    print('')


# The first question I'd like to answer:
# 
# Does non-flat topography have a detrimental effect on the good indicators of a thriving city? In other words, does the Topography type "Relatively Flat" have an advantage over the other three types when it comes to indicators of advanced technology, like "Advanced Drone Drive", "Artificial Muscles", "Forever Young", and all the other good-sounding advances?
# 
# To answer the question, we must first decide what are indicators of a thriving city. Looking at the column names, it seems that all the boolean type columns are upgrades, meaning that if you were playing the game, then you'd eventually want your civilization to have these capabilities. If that's the case, we can simply check whether the flat topography, ostensibly better areas of the map for building cities, have more True than False in these columns.

# In[ ]:


# get the names of all boolean type columns
bool_cols = df.select_dtypes(include=['bool']).columns


# Now that we have a list of all the columns of type bool, lets create a new column to hold the count of all the True values in those columns. Then we can check if the mean of all "Relatively Flat" counts is more than the mean counts of the other 3 types.

# In[ ]:


# a new column for the True counts of the bool columns in each row
df['bool True count'] = df.loc[:,bool_cols].sum(axis=1)
df.head(5)


# In[ ]:


# get the indexes of "Relatively Flat" rows
idx = np.where(df['Topography']=="Relatively Flat")
# the number of "Relatively Flat " entries
nb_rflat = len(idx[0])
# get the average boolean True count for all "Relatively Flat" entries
df.loc[idx]['bool True count'].values.sum()/nb_rflat


# Now, let's compare this value, 17, with the mean boolean True counts for the other three types of Topography entries. First up is "Steep":

# In[ ]:


idx = np.where(df['Topography']=="Steep")
nb_steep = len(idx[0])
df.loc[idx]['bool True count'].values.sum()/nb_steep


# Ok, ... that's weird, it's 17 too. Next let's find the mean counts for "Rough":

# In[ ]:


idx = np.where(df['Topography']=="Rough")
nb_rough = len(idx[0])
df.loc[idx]['bool True count'].values.sum()/nb_rough


# Whuuut? Is the mean for "Mountainous" also 17?

# In[ ]:


idx = np.where(df['Topography']=="Mountainous")
nb_mount = len(idx[0])
df.loc[idx]['bool True count'].values.sum()/nb_mount


# Are you kidding me? Each of the four Topography types has the same mean number of True values? There are 17 True values in every row in this 50901 rows set! These Trues are not all in the same columns either; you can simply print more rows to verify. I assume then that this value is dictated by the ruleset. Maybe I need to learn more about this game to understand why this is.

# In[ ]:




