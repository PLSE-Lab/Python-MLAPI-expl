#!/usr/bin/env python
# coding: utf-8

# # Santa`s Workshop Tour 2019!!!
# 
# 5000 families, 100 days, Santa get a huge problem, he need us, the Santa's knight, charge, to the workshop.
# 
# ## Load Lib, Load Data

# In[ ]:


import os,sys,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


fpath = '/kaggle/input/santa-2019-workshop-scheduling/family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

fpath = '/kaggle/input/santa-2019-workshop-scheduling/sample_submission.csv'
submission = pd.read_csv(fpath, index_col='family_id')


# In[ ]:


data.head()


# In[ ]:


submission.head()


# ## Something
# 
# - families:5000
# - days:100
# - round:125~300 people
# - cost rules:
#     - choice_0: no consolation gifts
#     - choice_1: one \$50 gift card to Santa's Gift Shop
#     - choice_2: one \$50 gift card, and 25% off Santa's Buffet (value \$9) for each family member
#     - choice_3: one \$100 gift card, and 25% off Santa's Buffet (value \$9) for each family member
#     - choice_4: one \$200 gift card, and 25% off Santa's Buffet (value \$9) for each family member
#     - choice_5: one \$200 gift card, and 50% off Santa's Buffet (value \$18) for each family member
#     - choice_6: one \$300 gift card, and 50% off Santa's Buffet (value \$18) for each family member
#     - choice_7: one \$300 gift card, and free Santa's Buffet (value \$36) for each family member
#     - choice_8: one \$400 gift card, and free Santa's Buffet (value \$36) for each family member
#     - choice_9: one \$500 gift card, and free Santa's Buffet (value \$36) for each family member, and 50% off North Pole Helicopter Ride tickets (value \$199) for each family member
#     - otherwise: one \$500 gift card, and free Santa's Buffet (value \$36) for each family member, and free North Pole Helicopter Ride tickets (value \$398) for each family member

# In[ ]:


data['cost_0'] = 0
data['cost_1'] = 50
data['cost_2'] = data[['choice_2','n_people']].apply(lambda row:50+9*row.n_people, axis=1)
data['cost_3'] = data[['choice_3','n_people']].apply(lambda row:100+9*row.n_people, axis=1)
data['cost_4'] = data[['choice_4','n_people']].apply(lambda row:200+9*row.n_people, axis=1)
data['cost_5'] = data[['choice_5','n_people']].apply(lambda row:200+18*row.n_people, axis=1)
data['cost_6'] = data[['choice_6','n_people']].apply(lambda row:300+18*row.n_people, axis=1)
data['cost_7'] = data[['choice_7','n_people']].apply(lambda row:300+36*row.n_people, axis=1)
data['cost_8'] = data[['choice_8','n_people']].apply(lambda row:400+36*row.n_people, axis=1)
data['cost_9'] = data[['choice_9','n_people']].apply(lambda row:500+36*row.n_people+199*row.n_people, axis=1)
data['cost_otherwise'] = data[['choice_9','n_people']].apply(lambda row:500+36*row.n_people+398*row.n_people, axis=1)


# ### algorithm
# 
# Idea: big family first, small family next, like throw rock to bottle, big first, small next, then you can full it.

# In[ ]:


data['visit_idx'] = -1
data['visit_day'] = -1
data['actual_cost'] = -1
workshop = {day:0 for day in range(1,101,1)}
data = data.sort_index(by=['n_people'],ascending=False)
for idx in range(len(data)):
    row = data.iloc[idx]
    checked = False
    for _choice in range(0,10,1):
        _idx = _choice
        _choice = 'choice_'+str(_choice)
        _day = row[_choice]
        if workshop[_day]+row['n_people']<300:
            row['visit_idx'] = _idx
            row['visit_day'] = _day
            row['actual_cost'] = row['cost_'+str(_idx)]
            workshop[_day] += row['n_people']
            checked = True
            break


# ### Otherwise

# In[ ]:


data['actual_cost'] = data[['visit_day','cost_otherwise','actual_cost']].apply(lambda row:row['cost_otherwise'] if row['visit_day']==-1 else row['actual_cost'], axis=1)
def illegal_day(row):
    if row['visit_day']!=-1:
        return row['visit_day']
    for day in range(1,101,1):
        if workshop[day]+row['n_people']<300:
            workshop[day] += row['n_people']
            return day
data['visit_day'] = data[['visit_day','n_people']].apply(illegal_day, axis=1)


# ### Check Result

# In[ ]:


workshop


# In[ ]:


data.actual_cost.sum()


# In[ ]:


data[data.visit_day==2].n_people.sum()


# ## Make Submission File

# In[ ]:


submission['assigned_day'] = data['visit_day'].tolist()
score = data.actual_cost.sum()
submission.to_csv(f'submission_{score}.csv')
print(f'Score: {score}')

