#!/usr/bin/env python
# coding: utf-8

# ## A Quick First Analysis 
# 
# To start, let's have a look at the most lines delivered by each charater in Deep Space Nine.
# 
# > From the processed dataset, generate counts of all lines spoken by each charater.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


all_series_lines=pd.read_json("../input/all_series_lines.json")


# In[ ]:


episodes=all_series_lines['DS9'].keys()


# In[ ]:


total_lines_counts={}
line_counts_by_episode={}
for i,ep in enumerate(episodes):
    episode="episode "+str(i)
    line_counts_by_episode[episode]={}
    if all_series_lines['DS9'][ep] is not np.NaN:
        for member in list(all_series_lines['DS9'][ep].keys()):
            line_counts_by_episode[episode][member]=len(all_series_lines['DS9'][ep][member])
            if member in total_lines_counts.keys():
                total_lines_counts[member]=total_lines_counts[member]+len(all_series_lines['DS9'][ep][member])
            else:
                total_lines_counts[member]=len(all_series_lines['DS9'][ep][member])


# In[ ]:


DS9_df=pd.DataFrame(list(total_lines_counts.items()), columns=['Character','No. of Lines'])
Top20=DS9_df.sort_values(by='No. of Lines', ascending=False).head(20)

Top20.plot.bar(x='Character',y='No. of Lines')
plt.show()


# In[ ]:




