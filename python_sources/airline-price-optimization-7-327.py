#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# This is notebook lists all my solutions to the [Airline Price Optimization Micro-Challenge](https://www.kaggle.com/general/62469). 

# In[ ]:


import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me


# ## Prelimanaries: Combinatorics
# Before getting started, I want to have an idea of the search space for this problem. How many ways can I sell `tickets_left` over `days_left`? Let's say I have `5` tickets, and `2` days to sell them. 
# * (0, 5)
# * (5, 0)
# * (1, 4)
# * (4, 1)
# * (3, 2)
# * (2, 3)
# 
# That gives me `6` ways to get to `5`. Cool, but is there a generic solution for this? I can frame this as a combinatorics problem where I'm looking for all the ways in which we can split `tickets_left` with `days_left - 1` dividers. 
# 
# Imagine the problem as a row of `5` one's. 
# `11111`
# 
# I can then split it into `2` groups by adding a divider somewhere along that row.<br>
# `|1111` (0, 5)<br>
# `1|1111` (1, 4)<br>
# `11|111` (2, 3)<br>
# `111|11` (3, 2)<br>
# $\dots$
# 
# From this point of view, the problem becomes: how many ways can I order `ticket_left` + `days_left - 1`, or $n + k$, items.
# 
# $f(n, k) = \frac{(n+k)!}{n!k!}$
# 
# Let's verify!

# In[ ]:


import numpy as np
def f(n, k): return np.math.factorial(n+k) // (np.math.factorial(n) * np.math.factorial(k))

def combos(n):
    """
    find all the combinations for 2 days
    """
    combos = 0
    items = set(range(n+1))
    for i in range(n+1):
        if (n - i) in items:
            combos += 1
    return combos

for i in range(5, 20):
    assert combos(i) == f(i, 1)


# So if we have `50` tickets and `7` days to sell them, it means there are $\frac{(56)!}{50!6!} = 32,468,436$ ways to do so!

# ## Baseline
# The baseline model provided by `Kaggle` sells `10` tickets every day. 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'def pricing_function(days_left, tickets_left, demand_level):\n    return demand_level - 10\nscore_me(pricing_function, sims_per_scenario=1000)')


# ## Method 0: Random
# What if we just randomly chose the price?

# In[ ]:


def pricing_function(days_left, tickets_left, demand_level):
    n_tickets = np.random.randint(0, tickets_left)
    return demand_level - n_tickets  
score_me(pricing_function, sims_per_scenario=1000)


# Naturally, that is a bad approach. 

# ##  Method 1: Heuristics

# Sell whatever the average number of tickets left is.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'def pricing_function(days_left, tickets_left, demand_level):\n    return demand_level - tickets_left // days_left\nscore_me(pricing_function, sims_per_scenario=1000)')


# This method isn't as fast as the baseline, but it made us an extra `$340` on average!
# 
# What if we took that a step further and only sold the average number of tickets left, when the demand is over 150?

# In[ ]:


get_ipython().run_cell_magic('time', '', 'def pricing_function(days_left, tickets_left, demand_level):\n    if days_left == 1:\n        return demand_level - tickets_left\n    if demand_level >= 150:\n        n_tickets =  tickets_left // days_left\n    else:\n        n_tickets = 0\n    return demand_level - n_tickets\n\nscore_me(pricing_function, sims_per_scenario=1000)')


# Now we've made `$905` above the baseline!

# ## Method 2: Heuristics + Probability
# 
# What if we know the chance that the demand level will go up, or down in the future? That should impact the number of tickets sold on a single day.  
# 
# We know from the problem statement that the demand level follows a `uniform distribution` between `100` and `200`. By calculating the inverse `CDF` of the `Uniform distribution` we can determine the chance that the next day we'll get a higher demand level. We can then use the `CDF` of the `Binomial distribution`, to calculate the chance that we'll see at least one day with a higher `demand_level` over `days_left`.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import scipy as sp\nimport scipy.stats    \nimport scipy.special\n\ndef pmf(n, k, p):\n    """\n    pmf of binomial distribution\n    """\n    return sp.special.comb(n, k) * (p ** k) * (1 - p) ** (n - k)\n\ndef cdf(n, k, p):\n    """\n    cdf of biomial distribution\n    """\n    return sum([pmf(n, i, p) for i in range(0, k+1)])\n\ndef chance_of_higher_demand_tomorrow(demand_level):\n    """\n    inverse cdf of uniform distritbuion\n    """\n    mn, mx = 100, 200\n    return 1 - (demand_level - mn) / (mx - mn)\n\ndef pricing_function(days_left, tickets_left, demand_level):\n    expected_demand = 150\n    p = chance_of_higher_demand_tomorrow(demand_level)    \n    chance_of_higher_demand = 1 - cdf(days_left, 1, p) # chance of having at least one success over {days_left} with prob {p}\n    \n    if days_left == 1:\n        n_tickets = tickets_left\n    elif chance_of_higher_demand <= 0.95:\n        n_tickets = tickets_left // days_left + ((1 - chance_of_higher_demand) / 3) * tickets_left\n\n    else:\n        n_tickets = 0\n    return demand_level - n_tickets\n\nscore_me(pricing_function, sims_per_scenario=1000)')


# ![](http://)Instead being lead by averages, here we're actually computing the likelihood that we'll get a higher demand in the future. This leads to a `$1,295` increase from the baseline! 
# 

# ---
# *This micro-challenge is from an exercise in an upcoming Optimization course on **[Kaggle Learn](https://www.kaggle.com/Learn?utm_medium=website&utm_source=kaggle.com&utm_campaign=micro+challenge+2018)**.  If you enjoyed this challenge and want to beef up your data science skills, you might enjoy our other courses.*
