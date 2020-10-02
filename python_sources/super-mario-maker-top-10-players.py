#!/usr/bin/env python
# coding: utf-8

# I created this work book as a challenge for myself to get the players with the most records from the Super Mario Maker for Wii U from the SMMnet dataset and then present the data.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from collections import Counter


# In[ ]:


# Read the data from a CSV into a pandas Dataframe
filepath = "../input/smmnet/records.csv"
data = pd.read_csv(filepath, index_col="player", sep="	")


# In[ ]:


# This stage is optional but it shows us the first 5 rows of data we have imported 
data.head()


# In[ ]:


# We set identify the 10 most common player names in the Dataframe which gives us out top 10 players
mc = Counter(data.index).most_common(10)

# We set the names and values variable to hold the information for our X & Y axis and then create the plot
names, values = map(list, zip(*mc))
indexes = np.arange(len(names))
plt.figure(figsize=(20,10))
plt.bar(indexes, values)
plt.xticks(indexes, names)

