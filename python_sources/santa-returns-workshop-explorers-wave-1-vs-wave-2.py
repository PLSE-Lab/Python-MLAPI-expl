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
# 1. 5000/6000(wave1/wave2) Families with variable number of members
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


# ## Wave 1

# In[ ]:


family_data_w1 = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')
sample_sub_w1 = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv', index_col='family_id')


# ## Wave 2

# In[ ]:


family_data_w2 = pd.read_csv("../input/santa-2019-revenge-of-the-accountants/family_data.csv", index_col='family_id')
sample_sub_w2 = pd.read_csv("../input/santa-2019-revenge-of-the-accountants/sample_submission.csv", index_col='family_id')


# #### New Faster cost function

# ## Wave 1 Cost Function

# In[ ]:


## from https://www.kaggle.com/nickel/250x-faster-cost-function-with-numba-jit
#prediction = sample_sub['assigned_day'].values
desired_w1 = family_data_w1.values[:, :-1]
family_size_w1 = family_data_w1.n_people.values
penalties_w1 = np.asarray([
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
    ] for n in range(family_size_w1.max() + 1)
])


# ## Wave 2 Cost Function

# In[ ]:


desired_w2 = family_data_w2.values[:, :-1]
family_size_w2 = family_data_w2.n_people.values
penalties_w2 = np.asarray([
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
    ] for n in range(family_size_w2.max() + 1)
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

# ## Wave 1

# In[ ]:


family_data_w1.shape


# In[ ]:


family_data_w2.shape


# In[ ]:


family_data_w1.head()


# In[ ]:


family_data_w2.head()


# #### The family_data that we are exploring represents each family in a row. So one family is one row here. Then each family has 10 preferences so 0 to 9 choices are given as columns. The number of people in each family are given by the column n_people. Each family is uniquely identified by a family_id. This is the data we will be using to help Santa!

# Let's check the data's health before we make any optimisation moves.

# In[ ]:


family_data_w1.isnull().sum()


# In[ ]:


family_data_w2.isnull().sum()


# No missing values!

# # What is the total number of people visiting?

# In[ ]:


family_data_w1.n_people.sum()


# In[ ]:


family_data_w2.n_people.sum()


# # If maximum people visit everyday how many days would this affair last?

# In[ ]:


family_data_w1.n_people.sum()/300


# In[ ]:


family_data_w2.n_people.sum()/300


# # How big are the families?

# ## Wave 1

# In[ ]:


## thanks to https://www.kaggle.com/chewzy/santa-finances-a-closer-look-at-the-costs

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

no_of_people = family_data_w1['n_people'].value_counts().sort_index()

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


# ## Wave 2

# In[ ]:



import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

no_of_people = family_data_w2['n_people'].value_counts().sort_index()

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


# ### Some small changes in the distribution are there.

# # Experiments with costs on the basis of choices

# ### What would the cost of assigning the same nth choice to each family be?

# ## Wave 1

# In[ ]:


assigned_days_choice_cost_w1 = [] 

sample_sub_w1['assigned_day'] = family_data_w1.choice_0
choice_0_cost, _ = jited_cost(sample_sub_w1['assigned_day'].values, desired_w1, family_size_w1,penalties_w1)
assigned_days_choice_cost_w1.append(choice_0_cost)
print("Choice 1 cost: {}".format(choice_0_cost))

sample_sub_w1['assigned_day'] = family_data_w1.choice_1
choice_1_cost, _ = jited_cost(sample_sub_w1['assigned_day'].values, desired_w1, family_size_w1,penalties_w1)
assigned_days_choice_cost_w1.append(choice_1_cost)
print("Choice 2 cost: {}".format(choice_1_cost))

sample_sub_w1['assigned_day'] = family_data_w1.choice_2
choice_2_cost, _ = jited_cost(sample_sub_w1['assigned_day'].values, desired_w1, family_size_w1,penalties_w1)
assigned_days_choice_cost_w1.append(choice_2_cost)
print("Choice 3 cost: {}".format(choice_2_cost))

sample_sub_w1['assigned_day'] = family_data_w1.choice_3
choice_3_cost, _ = jited_cost(sample_sub_w1['assigned_day'].values, desired_w1, family_size_w1,penalties_w1)
assigned_days_choice_cost_w1.append(choice_3_cost)
print("Choice 4 cost: {}".format(choice_3_cost))

sample_sub_w1['assigned_day'] = family_data_w1.choice_4
choice_4_cost, _ = jited_cost(sample_sub_w1['assigned_day'].values, desired_w1, family_size_w1,penalties_w1)
assigned_days_choice_cost_w1.append(choice_4_cost)
print("Choice 5 cost: {}".format(choice_4_cost))

sample_sub_w1['assigned_day'] = family_data_w1.choice_5
choice_5_cost, _ = jited_cost(sample_sub_w1['assigned_day'].values, desired_w1, family_size_w1,penalties_w1)
assigned_days_choice_cost_w1.append(choice_5_cost)
print("Choice 6 cost: {}".format(choice_5_cost))

sample_sub_w1['assigned_day'] = family_data_w1.choice_6
choice_6_cost, _ = jited_cost(sample_sub_w1['assigned_day'].values, desired_w1, family_size_w1,penalties_w1)
assigned_days_choice_cost_w1.append(choice_6_cost)
print("Choice 7 cost: {}".format(choice_6_cost))

sample_sub_w1['assigned_day'] = family_data_w1.choice_7
choice_7_cost, _ = jited_cost(sample_sub_w1['assigned_day'].values, desired_w1, family_size_w1,penalties_w1)
assigned_days_choice_cost_w1.append(choice_7_cost)
print("Choice 8 cost: {}".format(choice_7_cost))

sample_sub_w1['assigned_day'] = family_data_w1.choice_8
choice_8_cost, _ = jited_cost(sample_sub_w1['assigned_day'].values, desired_w1, family_size_w1,penalties_w1)
assigned_days_choice_cost_w1.append(choice_8_cost)
print("Choice 9 cost: {}".format(choice_8_cost))

sample_sub_w1['assigned_day'] = family_data_w1.choice_9
choice_9_cost, _ = jited_cost(sample_sub_w1['assigned_day'].values, desired_w1, family_size_w1,penalties_w1)
assigned_days_choice_cost_w1.append(choice_9_cost)
print("Choice 10 cost: {}".format(choice_9_cost))



# In[ ]:


plt.figure(figsize = (10,5))
_ =sns.lineplot(x = list(range(1,11)), y = assigned_days_choice_cost_w1,  marker = "*", markersize = 20, color = 'green')

plt.title('Penalty by Choice Number')
plt.xlabel("Choice Number")
plt.ylabel("Penalty")


# ### Choice 1 for all families together is causing the biggest loss to Santa by far!

# ## Wave 2

# In[ ]:


assigned_days_choice_cost_w2 = []
sample_sub_w2['assigned_day'] = family_data_w2.choice_0
choice_0_cost, _ = jited_cost(sample_sub_w2['assigned_day'].values, desired_w2, family_size_w2,penalties_w2)
assigned_days_choice_cost_w2.append(choice_0_cost)
print("Choice 1 cost: {}".format(choice_0_cost))

sample_sub_w2['assigned_day'] = family_data_w2.choice_1
choice_1_cost, _ = jited_cost(sample_sub_w2['assigned_day'].values, desired_w2, family_size_w2,penalties_w2)
assigned_days_choice_cost_w2.append(choice_1_cost)
print("Choice 2 cost: {}".format(choice_1_cost))

sample_sub_w2['assigned_day'] = family_data_w2.choice_2
choice_2_cost, _ = jited_cost(sample_sub_w2['assigned_day'].values, desired_w2, family_size_w2,penalties_w2)
assigned_days_choice_cost_w2.append(choice_2_cost)
print("Choice 3 cost: {}".format(choice_2_cost))

sample_sub_w2['assigned_day'] = family_data_w2.choice_3
choice_3_cost, _ = jited_cost(sample_sub_w2['assigned_day'].values, desired_w2, family_size_w2,penalties_w2)
assigned_days_choice_cost_w2.append(choice_3_cost)
print("Choice 4 cost: {}".format(choice_3_cost))

sample_sub_w2['assigned_day'] = family_data_w2.choice_4
choice_4_cost, _ = jited_cost(sample_sub_w2['assigned_day'].values, desired_w2, family_size_w2,penalties_w2)
assigned_days_choice_cost_w2.append(choice_4_cost)
print("Choice 5 cost: {}".format(choice_4_cost))

sample_sub_w2['assigned_day'] = family_data_w2.choice_5
choice_5_cost, _ = jited_cost(sample_sub_w2['assigned_day'].values, desired_w2, family_size_w2,penalties_w2)
assigned_days_choice_cost_w2.append(choice_5_cost)
print("Choice 6 cost: {}".format(choice_5_cost))

sample_sub_w2['assigned_day'] = family_data_w2.choice_6
choice_6_cost, _ = jited_cost(sample_sub_w2['assigned_day'].values, desired_w2, family_size_w2,penalties_w2)
assigned_days_choice_cost_w2.append(choice_6_cost)
print("Choice 7 cost: {}".format(choice_6_cost))

sample_sub_w2['assigned_day'] = family_data_w2.choice_7
choice_7_cost, _ = jited_cost(sample_sub_w2['assigned_day'].values, desired_w2, family_size_w2,penalties_w2)
assigned_days_choice_cost_w2.append(choice_7_cost)
print("Choice 8 cost: {}".format(choice_7_cost))

sample_sub_w2['assigned_day'] = family_data_w2.choice_8
choice_8_cost, _ = jited_cost(sample_sub_w2['assigned_day'].values, desired_w2, family_size_w2,penalties_w2)
assigned_days_choice_cost_w2.append(choice_8_cost)
print("Choice 9 cost: {}".format(choice_8_cost))

sample_sub_w2['assigned_day'] = family_data_w2.choice_9
choice_9_cost, _ = jited_cost(sample_sub_w2['assigned_day'].values, desired_w2, family_size_w2,penalties_w2)
assigned_days_choice_cost_w2.append(choice_9_cost)
print("Choice 10 cost: {}".format(choice_9_cost))



# In[ ]:


plt.figure(figsize = (10,5))
_ =sns.lineplot(x = list(range(1,11)), y = assigned_days_choice_cost_w2,  marker = "*", markersize = 20, color = 'green')

plt.title('Penalty by Choice Number')
plt.xlabel("Choice Number")
plt.ylabel("Penalty")


# ### Choice 2 for all families together is causing the biggest loss to Santa! That's not the same as last time.

# ## More Updates coming up soon!
