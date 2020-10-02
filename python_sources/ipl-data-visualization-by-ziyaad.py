#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


data = pd.read_excel("../input/DIM_MATCH.xlsx")

data.head()


# In[ ]:


info = data.ManOfMach.value_counts()[0:10]

fig,ax = plt.subplots(figsize = (15,6))

ax.bar(info.keys(),info)

plt.title("Man of the Match")
plt.xlabel("Player Name")
plt.ylabel("No of times")

plt.show()




    


# In[ ]:


data.isnull().sum()

data=data.dropna(axis=0,how="any")

data.isnull().sum()


# In[ ]:


data.count()


# In[ ]:


data.head()

info = data.Season_Year.value_counts()

fig,ax = plt.subplots(figsize = (12,6))

ax.bar(info.keys(),info, color="c", alpha=.3, width=.5)

plt.xlabel("Year")
plt.ylabel("Matches Played")

plt.show()


# In[ ]:


info = data.Venue_Name.value_counts()

fig,ax = plt.subplots(figsize = (15,6))

ax.bar(info.keys(),info)

ax.set_xticklabels(info.keys())
plt.tight_layout()

fig.autofmt_xdate()

plt.show()


# In[ ]:


data.head()

info = data.match_winner.value_counts()    

fig,ax = plt.subplots()

ax = plt.pie(info,labels=info.keys(), autopct = '%1.1f%%', radius=2)


plt.show()

