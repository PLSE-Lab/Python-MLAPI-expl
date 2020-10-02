#!/usr/bin/env python
# coding: utf-8

#  <h1>Finding a king with most wins</h1>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/battles.csv")
df.head(10)


# In[ ]:


kings = pd.Series(df["attacker_king"].append(df["defender_king"].reset_index(drop=True)).unique())
kings.dropna(inplace = True)
kings


# In[ ]:


outcomes = df[["name", "attacker_king", "defender_king", "attacker_outcome"]]
outcomes.head(3)


# In[ ]:


attwin = outcomes[outcomes.attacker_outcome == "win"]["attacker_king"]
attwin.dropna(inplace = True)
attwin


# In[ ]:


defwin = outcomes[outcomes.attacker_outcome == "loss"]["defender_king"]
defwin.dropna(inplace = True)
defwin


# In[ ]:


winnerkings = defwin.append(attwin).reset_index(drop=True)
winnerkings


# In[ ]:


winnerkings.value_counts()


# In[ ]:



kingsscores = pd.DataFrame({"kings_name":[], "number_of_wins": []})
i = 0
for king in kings:
    if king in winnerkings.value_counts():
        kingsscores.loc[df.index[i],["kings_name"]] = king
        kingsscores.loc[df.index[i],["number_of_wins"]] = winnerkings.value_counts()[king]
    else:
        kingsscores.loc[df.index[i],["kings_name"]] = king
        kingsscores.loc[df.index[i],["number_of_wins"]] = 0
    i +=1

kingsscores


# In[ ]:


plt.figure(dpi = 200)
plt.bar(kingsscores["kings_name"], kingsscores["number_of_wins"], 
        edgecolor = "k",
       )
plt.xticks(kings, rotation = "vertical")
plt.grid()

