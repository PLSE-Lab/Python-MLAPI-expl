#!/usr/bin/env python
# coding: utf-8

# Thanks for Yury Dzerin 's notebook . 
# https://www.kaggle.com/deyury/starter-notebook-greedy-initialization/output
# 
# I have tried some ideas to further improve the score. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


fpath = '/kaggle/input/santa-workshop-tour-2019/family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

fpath = '/kaggle/input/santa-workshop-tour-2019/sample_submission.csv'
submission = pd.read_csv(fpath, index_col='family_id')

# lets upload Yury's ouput file to further save time
fpath = '/kaggle/input/starter-notebook-greedy-initialization/submission_430074.7764025667.csv'
greedy = pd.read_csv(fpath, index_col='family_id')


# In[ ]:


# lets find all rows which are assigned to no choices ( otherwise- most costly )and try to assign choice 0-9 
others_list = []
for ind,row in greedy.iterrows():
    if row['assigned_day'] not in data.iloc[ind,:-1].to_list():
        others_list.append(ind)
        
        


# In[ ]:


len(others_list)


# There are 101 family id which are assigned to otherwise dates. Lets try to assign them better dates and see if score improves . I am not running for entire data to save time.

# In[ ]:


family_size_dict = data[['n_people']].to_dict()['n_people']

cols = [f'choice_{i}' for i in range(10)]
choice_dict = data[cols].to_dict()

N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

# from 100 to 1
days = list(range(N_DAYS,0,-1))


# In[ ]:


def cost_function(prediction):

    penalty = 0

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = {k:0 for k in days}
    
    # Looping over each family; d is the day for each family f
    for f, d in enumerate(prediction):

        # Using our lookup dictionaries to make simpler variable names
        n = family_size_dict[f]
        choice_0 = choice_dict['choice_0'][f]
        choice_1 = choice_dict['choice_1'][f]
        choice_2 = choice_dict['choice_2'][f]
        choice_3 = choice_dict['choice_3'][f]
        choice_4 = choice_dict['choice_4'][f]
        choice_5 = choice_dict['choice_5'][f]
        choice_6 = choice_dict['choice_6'][f]
        choice_7 = choice_dict['choice_7'][f]
        choice_8 = choice_dict['choice_8'][f]
        choice_9 = choice_dict['choice_9'][f]

        # add the family member count to the daily occupancy
        daily_occupancy[d] += n

        # Calculate the penalty for not getting top preference
        if d == choice_0:
            penalty += 0
        elif d == choice_1:
            penalty += 50
        elif d == choice_2:
            penalty += 50 + 9 * n
        elif d == choice_3:
            penalty += 100 + 9 * n
        elif d == choice_4:
            penalty += 200 + 9 * n
        elif d == choice_5:
            penalty += 200 + 18 * n
        elif d == choice_6:
            penalty += 300 + 18 * n
        elif d == choice_7:
            penalty += 300 + 36 * n
        elif d == choice_8:
            penalty += 400 + 36 * n
        elif d == choice_9:
            penalty += 500 + 36 * n + 199 * n
        else:
            penalty += 500 + 36 * n + 398 * n

    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    for _, v in daily_occupancy.items():
        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):
            penalty += 100000000

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)
    
    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
        yesterday_count = today_count

    penalty += accounting_cost

    return penalty


# In[ ]:


# Lets try to improve score by working on family with otherwise date assigned.
best = greedy['assigned_day'].tolist()
start_score = 430074.7764025667

new = best.copy()
# loop over each family
for _, fam_id in enumerate(others_list):
    # loop over each family choice
    for pick in range(10):
        day = choice_dict[f'choice_{pick}'][fam_id]
        temp = new.copy()
        temp[fam_id] = day # add in the new pick
        if cost_function(temp) < start_score:
            new = temp.copy()
            start_score = cost_function(new)

submission['assigned_day'] = new
score = cost_function(new)

#submission.to_csv(f'submission_{score}.csv')
print(f'Score: {score}')


# Score didnt improve. Lets try for other options like choice9,8 ..etc.

# In[ ]:


list_9 = []
for ind,row in submission.iterrows():
    if row['assigned_day'] == data.iloc[ind,-1]:
        list_9.append(ind)


# In[ ]:


len(list_9)


# In[ ]:


# Lets try to improve score by working on family with choice9 assigned
best = greedy['assigned_day'].tolist()
start_score = 430074.7764025667

new = best.copy()
# loop over each family
for _, fam_id in enumerate(list_9):
    # loop over each family choice
    for pick in range(10):
        day = choice_dict[f'choice_{pick}'][fam_id]
        temp = new.copy()
        temp[fam_id] = day # add in the new pick
        if cost_function(temp) < start_score:
            new = temp.copy()
            start_score = cost_function(new)

submission['assigned_day'] = new
score = cost_function(new)

#submission.to_csv(f'submission_{score}.csv')
print(f'Score: {score}')


# It improved score . Lets try with other choices . We can easily put in loop and play more with it . 
# Also , we can sort by family size & day.

# In[ ]:


list_8 = []
for ind,row in submission.iterrows():
    if row['assigned_day'] == data.iloc[ind,-2]:
        list_8.append(ind)


# In[ ]:


# Lets try to improve score by working on family with choice8 assigned



best = submission['assigned_day'].tolist()
start_score = cost_function(best)

new = best.copy()
# loop over each family
for _, fam_id in enumerate(list_8):
    # loop over each family choice
    for pick in range(10):
        day = choice_dict[f'choice_{pick}'][fam_id]
        temp = new.copy()
        temp[fam_id] = day # add in the new pick
        if cost_function(temp) < start_score:
            new = temp.copy()
            start_score = cost_function(new)

submission['assigned_day'] = new
score = cost_function(new)

#submission.to_csv(f'submission_{score}.csv')
print(f'Score: {score}')


# In[ ]:


list_7 = []
for ind,row in submission.iterrows():
    if row['assigned_day'] == data.iloc[ind,-3]:
        list_7.append(ind)


# In[ ]:


len(list_7)


# In[ ]:



best = submission['assigned_day'].tolist()
start_score = cost_function(best)

new = best.copy()
# loop over each family
for _, fam_id in enumerate(list_7):
    # loop over each family choice
    for pick in range(10):
        day = choice_dict[f'choice_{pick}'][fam_id]
        temp = new.copy()
        temp[fam_id] = day # add in the new pick
        if cost_function(temp) < start_score:
            new = temp.copy()
            start_score = cost_function(new)

submission['assigned_day'] = new
score = cost_function(new)

#submission.to_csv(f'submission_{score}.csv')
print(f'Score: {score}')


# In[ ]:


submission.to_csv(f'submission_{score}.csv')


# I will not do it anymore . We can easily try this further by putting different choices (startng from 1) in loop, also sort days and family size since they affect cost.
# 
# 

# In[ ]:





# In[ ]:



