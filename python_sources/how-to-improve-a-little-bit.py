#!/usr/bin/env python
# coding: utf-8

# Just a dumb heuristic to slightly improve existing solutions.
# 
# Thanks to https://www.kaggle.com/jazivxt/using-a-baseline for the starting solution and to https://www.kaggle.com/nickel/santa-s-2019-fast-pythonic-cost-23-s for the cost function.

# In[ ]:


import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[ ]:


data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')
family_size = data.n_people.values


# ## Define cost function

# In[ ]:


penalties = np.asarray([
    [
        0,
        50,
        50 + 9 * n,
        100 + 9 * n,
        200 + 9 * n,
        200 + 18 * n,
        300 + 18 * n,
        300 + 36 * n,
        400 + 36 * n,
        500 + 36 * n + 199 * n,
        500 + 36 * n + 398 * n
    ] for n in range(family_size.max() + 1)
])
cost_matrix = np.concatenate(data.n_people.apply(lambda n: np.repeat(penalties[n, 10], 100).reshape(1, 100)))
for fam in data.index:
    for choice_order, day in enumerate(data.loc[fam].drop("n_people")):
        cost_matrix[fam, day - 1] = penalties[data.loc[fam, "n_people"], choice_order]
        
        
# adding fastmath=True as recomended by CFDAero
@njit(fastmath=True)
def cost_function(prediction, family_size, cost_matrix):
    N_DAYS = cost_matrix.shape[1]
    MAX_OCCUPANCY = 300
    MIN_OCCUPANCY = 125
    penalty = 0
    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int64)
    for i, (pred, n) in enumerate(zip(prediction, family_size)):
        daily_occupancy[pred - 1] += n
        penalty += cost_matrix[i, pred - 1]

    accounting_cost = 0
    n_low = 0
    n_high = 0
    daily_occupancy[-1] = daily_occupancy[-2]
    for day in range(N_DAYS):
        n_next = daily_occupancy[day + 1]
        n = daily_occupancy[day]
        n_high += (n > MAX_OCCUPANCY) 
        n_low += (n < MIN_OCCUPANCY)
        diff = abs(n - n_next)
        accounting_cost += max(0, (n-125.0) / 400.0 * n**(0.5 + diff / 50.0))

    #print(penalty, accounting_cost)
    penalty += accounting_cost
    
    
    if n_low>0 or n_high>0: penalty = np.inf
    return np.asarray([penalty, n_low, n_high, np.round(100*accounting_cost/penalty,3)])

get_cost = lambda prediction: cost_function(prediction, family_size, cost_matrix)


# ## "Greedy brute force"

# In[ ]:


get_ipython().system('ls /kaggle/input')


# In[ ]:


# select an intial solution
prediction = pd.read_csv("/kaggle/input/using-a-baseline/submission_72107.42009587162.csv").assigned_day.values


# In[ ]:


l = get_cost(prediction)[0]
print(l)


# In[ ]:


# try moving one family at a time
loss = [l]
while(len(loss)>0):
    loss = []
    for f in tqdm(range(5000)):
        for day in range(1,101):
            tmp = prediction.copy()
            tmp[f] = day
            new_l = get_cost(tmp)[0]
            if new_l<l:
                prediction = tmp
                l = new_l
                loss.append(l)
                print(l)

    plt.plot(loss)
    plt.show()

    print(l)


# In[ ]:


# swap pairs of families
loss = []
for f1 in tqdm(range(4999)):
    for f2 in range(f1,5000):
        tmp = prediction.copy()
        tmp[f1],tmp[f2] = tmp[f2],tmp[f1]
        new_l = get_cost(tmp)[0]
        if new_l<l:
            prediction = tmp
            l = new_l
            loss.append(l)
            print(l)

plt.plot(loss)
plt.show()

print(l)


# In[ ]:


best = get_cost(prediction)[0]


# In[ ]:


prediction = pd.Series(prediction, name="assigned_day")
prediction.index.name = "family_id"
prediction.to_csv("submission_"+str(best)+".csv", index=True, header=True)


# In[ ]:




