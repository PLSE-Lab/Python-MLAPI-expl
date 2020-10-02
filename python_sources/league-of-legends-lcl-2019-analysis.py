#!/usr/bin/env python
# coding: utf-8

# ### Read Data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/kaggle/input/league-of-legends-lcl-2019/lcl_match_history.csv')


# In[ ]:


data.head(10)


# ### Integrity Checking

# In[ ]:


data.isna().sum()


# Column MVP has NA values, as for me, I think we can't do anything with this except fill NA values from YouTube replays for every game.

# In[ ]:


data.duplicated().sum()


# #### Check Duplicates

# In[ ]:


bool_series = data.duplicated(keep = False) 

data[bool_series] 


# Happily, there is no duplicates in this dataset.

# ### Analysis

# I want to know the unrest for both red and blue sides and which winrate is bigger, but as we can see in the **Blue** and the **Red** column table have had just shortcuts for the full team names, we need to create dust and replace shortcuts with full names to handle this.

# In[ ]:


data['Team1'].value_counts()


# In[ ]:


data['Red'].value_counts()


# In[ ]:


data['Blue'].value_counts()


# In[ ]:


shortcut_dict = { 'GMB':'Gambit Esports', 'EPG':'Elements Pro Gaming', 'M19':'M19', 'VS':'Vaevictis eSports', 'RoX':'RoX', 'VEG':'Vega Squadron', 'DA':'Dragon Army', 'UOL':'Unicorns Of Love', 'TJ': 'Team Just'}


# In[ ]:


def replace_shortcuts(row):
    row['Red'] = shortcut_dict[row['Red']]
    row['Blue'] = shortcut_dict[row['Blue']]
    return row
data = data.apply(lambda row: replace_shortcuts(row), axis=1)


# In[ ]:


data


# Finally, we replaced all shortcuts, now we can start to plot chart!

# In[ ]:


sides = ['Blue', 'Red']
def count_win_on_side(row):
    if (row['Winner'] == row['Blue']):
        return pd.Series([1, 0], sides)
    else:
        return pd.Series([0, 1], sides)

sides_data = data.apply(lambda row: count_win_on_side(row), axis=1).mean()
sides_data


# In[ ]:


sides_data.apply(lambda row: '{:.2%}'.format(row))


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 7), subplot_kw=dict(aspect="equal"))

colors = ['#29b5ce', '#e03364']
plt.pie(sides_data, colors=colors, labels=sides)

ax.set_title('Distribution of the winning percentage by side', pad=20)
plt.axis('equal')
plt.show()


# As we can see, teams on Blue side have more unrest than Red side. Maybe Riot Games need to bugfix this?

# In[ ]:




