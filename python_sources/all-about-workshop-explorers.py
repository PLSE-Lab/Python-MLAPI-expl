#!/usr/bin/env python
# coding: utf-8

# ## Santa Business!

# ## Santa's special surprise:
# 
# 1. Tours to his workshop from 100 days before Christmas.
# 2. Extra perks for families that don't get their preference.
# 3. Ten preferences for visit to each family.
# 
# ## Santa's constraints:
# 
# 1. 5000 Families with variable number of members
# 2. 100 Days only
# 3. Cost attached to each family's visit must be limited
# 4. Total number of people (not families) attending the workshop each day must be between 125 - 300
# 5. Every family must be scheduled for one and only one assigned_day.
# 
# ## Let's lessen Santa's Workload
# 
# Santa's accountants have devised a formula of how cost would be calculated which is given below. We need to minimise this cost. This sort of a problem where we need to minimise or maximise a function's value output is known as Optimisation problem in Mathematics

# #### Simple Cost Function

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


family_data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')
sample_sub = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv', index_col='family_id')


# In[ ]:


family_size_dict = family_data[['n_people']].to_dict()['n_people']

cols = [f'choice_{i}' for i in range(10)]
choice_dict = family_data[cols].to_dict()

N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

# from 100 to 1
days = list(range(N_DAYS,0,-1))

def old_cost_function(prediction):

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


# #### New Faster cost function

# In[ ]:


## from https://www.kaggle.com/nickel/250x-faster-cost-function-with-numba-jit
#prediction = sample_sub['assigned_day'].values
desired = family_data.values[:, :-1]
family_size = family_data.n_people.values
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


# In[ ]:


## from https://www.kaggle.com/nickel/250x-faster-cost-function-with-numba-jit
from numba import njit

@njit()
def jited_cost(prediction, desired, family_size, penalties):
    N_DAYS = 100
    MAX_OCCUPANCY = 300
    MIN_OCCUPANCY = 125
    penalty = 0
    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int64)
    for i in range(len(prediction)):
        n = family_size[i]
        pred = prediction[i]
        n_choice = 0
        for j in range(len(desired[i])):
            if desired[i, j] == pred:
                break
            else:
                n_choice += 1
        
        daily_occupancy[pred - 1] += n
        penalty += penalties[n, n_choice]

    accounting_cost = 0
    n_out_of_range = 0
    daily_occupancy[-1] = daily_occupancy[-2]
    for day in range(N_DAYS):
        n_next = daily_occupancy[day + 1]
        n = daily_occupancy[day]
        n_out_of_range += (n > MAX_OCCUPANCY) or (n < MIN_OCCUPANCY)
        diff = abs(n - n_next)
        accounting_cost += max(0, (n-125.0) / 400.0 * n**(0.5 + diff / 50.0))

    penalty += accounting_cost
    return np.asarray([penalty, n_out_of_range])


# ## How do we help Santa?

# Let's look at what the data is telling us first!

# In[ ]:


family_data.shape


# In[ ]:


family_data.head()


# #### The family_data that we are exploring represents each family in a row. So one family is one row here. Then each family has 10 preferences so 0 to 9 choices are given as columns. The number of people in each family are given by the column n_people. Each family is uniquely identified by a family_id. This is the data we will be using to help Santa!

# Let's check the data's health before we make any optimisation moves.

# In[ ]:


family_data.isnull().sum()


# No missing values!

# #### What is the total number of people visiting?

# In[ ]:


family_data.n_people.sum()


# #### If maximum people visit everyday how many days would this affair last?

# In[ ]:


family_data.n_people.sum()/300


# #### How big are the families?

# In[ ]:


## thanks to https://www.kaggle.com/chewzy/santa-finances-a-closer-look-at-the-costs

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

no_of_people = family_data['n_people'].value_counts().sort_index()

plt.figure(figsize=(14,6))
ax = sns.barplot(x=no_of_people.index, y=no_of_people.values)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}\n({p.get_height() / sum(no_of_people) * 100:.1f}%)', 
                xy=(p.get_x() + p.get_width()/2., p.get_height()), ha='center', xytext=(0,5), textcoords='offset points')
    
ax.set_ylim(0, 1.1*max(no_of_people))
plt.xlabel('Number of people in family', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Family Size Distribution', fontsize=20)
plt.show()


# #### Most prefered days

# In[ ]:


choice_0 = family_data.choice_0.value_counts().sort_index()
choice_9 = family_data.choice_9.value_counts().sort_index()


# In[ ]:


plt.figure(figsize=(20,10))
_ = sns.barplot(x=choice_0.index, y=choice_0.values)


# In[ ]:


plt.figure(figsize=(20,10))
_ = sns.barplot(x=choice_9.index, y=choice_9.values)


# ### Experiments with costs on the basis of choices

# In[ ]:


## from https://www.kaggle.com/inversion/santa-s-2019-starter-notebook
# Start with the sample submission values
best = sample_sub['assigned_day'].values
start_score = old_cost_function(best)


# In[ ]:


start_score


# #### Comparison of new cost function output with the old one

# In[ ]:


new_start_score, errors = jited_cost(best, desired, family_size, penalties)


# #### Errors - count of the number of times the assigned days caused the condition to go out of bounds

# In[ ]:



errors


# In[ ]:


new_start_score


# ## Comes out to be the same. Perfect let's use the faster cost function for further exploration.

# #### What would the cost of assigning the same nth choice to each family be?

# In[ ]:


assigned_days_choice_cost = [] 


# In[ ]:


sample_sub['assigned_day'] = family_data.choice_0
choice_0_cost, _ = jited_cost(sample_sub['assigned_day'].values, desired, family_size,penalties)
assigned_days_choice_cost.append(choice_0_cost)
choice_0_cost


# In[ ]:


sample_sub['assigned_day'] = family_data.choice_1
choice_1_cost, _ = jited_cost(sample_sub['assigned_day'].values, desired, family_size,penalties)
assigned_days_choice_cost.append(choice_1_cost)
choice_1_cost


# In[ ]:


sample_sub['assigned_day'] = family_data.choice_2
choice_2_cost, _ = jited_cost(sample_sub['assigned_day'].values, desired, family_size,penalties)
assigned_days_choice_cost.append(choice_2_cost)
choice_2_cost


# In[ ]:


sample_sub['assigned_day'] = family_data.choice_3
choice_3_cost, _ = jited_cost(sample_sub['assigned_day'].values, desired, family_size,penalties)
assigned_days_choice_cost.append(choice_3_cost)
choice_3_cost


# In[ ]:


sample_sub['assigned_day'] = family_data.choice_4
choice_4_cost, _ = jited_cost(sample_sub['assigned_day'].values, desired, family_size,penalties)
assigned_days_choice_cost.append(choice_4_cost)
choice_4_cost


# In[ ]:


sample_sub['assigned_day'] = family_data.choice_5
choice_5_cost, _ = jited_cost(sample_sub['assigned_day'].values, desired, family_size,penalties)
assigned_days_choice_cost.append(choice_5_cost)
choice_5_cost


# In[ ]:


sample_sub['assigned_day'] = family_data.choice_6
choice_6_cost, _ = jited_cost(sample_sub['assigned_day'].values, desired, family_size,penalties)
assigned_days_choice_cost.append(choice_6_cost)
choice_6_cost


# In[ ]:


sample_sub['assigned_day'] = family_data.choice_7
choice_7_cost, _ = jited_cost(sample_sub['assigned_day'].values, desired, family_size,penalties)
assigned_days_choice_cost.append(choice_7_cost)
choice_7_cost


# In[ ]:


sample_sub['assigned_day'] = family_data.choice_8
choice_8_cost, _ = jited_cost(sample_sub['assigned_day'].values, desired, family_size,penalties)
assigned_days_choice_cost.append(choice_8_cost)
choice_8_cost


# In[ ]:


sample_sub['assigned_day'] = family_data.choice_9
choice_9_cost, _ = jited_cost(sample_sub['assigned_day'].values, desired, family_size,penalties)
assigned_days_choice_cost.append(choice_9_cost)
choice_9_cost


# In[ ]:


plt.figure(figsize = (10,5))
_ =sns.lineplot(x = list(range(1,11)), y = assigned_days_choice_cost,  marker = "*", markersize = 20, color = 'green')

plt.title('Penalty by Choice Number')
plt.xlabel("Choice Number")
plt.ylabel("Penalty")


# ### Choice 1 for all families together is causing the biggest loss to Santa by far!

# ## More Updates coming up soon!

# In[ ]:




