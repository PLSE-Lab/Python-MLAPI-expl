#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from itertools import permutations
get_ipython().run_line_magic('matplotlib', 'inline')

MIN_OCCUPANCY = 125
MAX_OCCUPANCY = 300
df = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col=0)
CHOICES = [c for c in df.columns if c.startswith('choice')]


# # Explore Family Size Distribution
# 
# Even though the dataset shows the distribution of family sizes, it is good to validate that our data looks right.

# In[ ]:


sizes = Counter(df.n_people)
print(sizes)
df.n_people.hist(bins=7, range=(1.5, 8.5))


# # Explore distribution of preferred days
# 
# To get a sense of day preferences, it would be good to see the distribution of each day of preference. We could take a historgram of each column, but that only tells you how many families would prefer that date and not how many people would be there on that date. Since not all families are the same size, this allows us to determine what days can be filled by first choice days.
# 
# There is a clear pattern to the data where there are three days of high popularity followed by four days of lower priority. Presumably, the three days are weekends.

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=5, sharex='all', sharey='all', figsize=(20, 4))
for col, ax in zip(CHOICES, axes.flatten()):
    df[[col, 'n_people']].groupby(col).sum().plot(ax=ax)
    ax.plot((0, 100), (125, 125), c='r', ls=':')
    ax.plot((0, 100), (300, 300), c='r', ls=':')
    ax.set_title(col)


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all', sharey='all', figsize=(20, 4))
for i, col in enumerate(CHOICES):
    df[[col, 'n_people']].groupby(col).sum().plot(ax=ax, label=i)
ax.plot((0, 100), (125, 125), c='r', ls=':')
ax.plot((0, 100), (300, 300), c='r', ls=':')
ax.legend()


# In[ ]:


# Check to see if there's an obvious correlation between family size and priority choice
plt.subplots(figsize=(20, 4))
plt.scatter(df.choice_0, df.n_people)


# # Gain Understanding of Penalties
# 
# The compensation package is pretty easy to understand, but the accounting penalty is not as straight forward.
# 
# The heatmap below shows that there is a steep penalty for a sudden drop in attendance, but not much of a problem for a sudden increase. Unfortunately, no money is gained for having more people than the day before.

# In[ ]:


def accounting_penalty(nd0, nd1):
    return max(0, (
        ((nd0 - 125) / 400)
        * nd0
        * (0.5 + ((nd0 - nd1) / 50))
    ))


# In[ ]:


acc_p_df = pd.DataFrame([*permutations(range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1), 2)], columns=['nd0', 'nd1'])
acc_p_df['penalty'] = acc_p_df.apply(lambda row: accounting_penalty(row.nd0, row.nd1), axis=1)
ax = sns.heatmap(acc_p_df.pivot(index='nd1', columns='nd0'))
ax.invert_yaxis()


# In[ ]:


def consolation(df: pd.DataFrame, family: int, day: int):
    members = df.loc[family, 'n_people']
    try:
        choice = int(np.where(df.loc[family].values.flatten() == day)[0])
    except TypeError:
        choice = 'x'
    cost = {
        0: 0,
        1: 50,
        2: 50 + 9 * members,
        3: 100 + 9 * members,
        4: 200 + 9 * members,
        5: 200 + 18 * members,
        6: 300 + 18 * members,
        7: 300 + 36 * members,
        8: 400 + 36 * members,
        9: 500 + 135 * members,
        'x':  500 + 434 * members,
    }
    
    print(members, choice)
    return cost[choice]


# In[ ]:




