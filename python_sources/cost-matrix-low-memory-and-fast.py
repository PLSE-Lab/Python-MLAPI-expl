#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numba import njit
import pickle
import os


# In[ ]:


data = pd.read_csv('../input/santa-workshop-tour-2019/family_data.csv', index_col='family_id', dtype=np.uint16)


# In[ ]:


@njit()
def preference_cost(n,c):
    return 450-100*(c<9)-100*(c<8)-100*(c<6)-100*(c<4)-50*(c<3)+50*(c>0) +            (36-(c<7)*18-(c<5)*9-(c<2)*9+199*(c>8)+199*(c>9))*n

cost_matrix = np.zeros([5000, 100], dtype=np.uint32)

for i in range(5000):
    tmp = data.loc[i].values
    n_people = tmp[-1]
    cols = tmp[:-1]
    #set all costs into other choice first
    cost_matrix[i,:] = preference_cost(n_people, 10)
    #replace costs for top 10 choices
    for choice,col in enumerate(cols):
        cost_matrix[i, col-1] = preference_cost(n_people, choice)
        
with open('cost_matrix.pickle', 'wb') as handle:
    pickle.dump(cost_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

# load the matrix pickle
# with open('cost_matrix.pickle', 'rb') as handle:
#     cost_matrix = pickle.load(handle)
# handle.close()


# In[ ]:


@njit() # https://www.kaggle.com/nickel/250x-faster-cost-function-with-numba-jit
def accounting_cost(prediction, preference, family_size):
    N_DAYS = 100
    MAX_OCCUPANCY = 300
    MIN_OCCUPANCY = 125
    penalty = 0
    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int64)
    for i in range(len(prediction)):
        n = family_size[i]
        pred = prediction[i]
        n_choice = 0
        for j in range(len(preference[i])):
            if preference[i, j] == pred:
                break
            else:
                n_choice += 1

        daily_occupancy[pred - 1] += n
        penalty += preference_cost(n, n_choice)

    acc_cost = 0
    n_out_of_range = 0
    daily_occupancy[-1] = daily_occupancy[-2]
    for day in range(N_DAYS):
        n_next = daily_occupancy[day + 1]
        n = daily_occupancy[day]
        n_out_of_range += (n > MAX_OCCUPANCY) or (n < MIN_OCCUPANCY)
        diff = abs(n - n_next)
        acc_cost += max(0, (n-125.0) / 400.0 * n**(0.5 + diff / 50.0))

    penalty += acc_cost
    return np.asarray([penalty, n_out_of_range])


# In[ ]:


submission = pd.read_csv('../input/santa-workshop-tour-2019/sample_submission.csv', index_col='family_id', dtype=np.uint16)

prediction = submission['assigned_day'].values
preference = data.values[:, :-1]
family_size = data.n_people.values

score, errors = accounting_cost(prediction, preference, family_size)
print('score =',score)

