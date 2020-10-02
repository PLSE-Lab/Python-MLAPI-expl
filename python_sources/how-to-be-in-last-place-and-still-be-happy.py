#!/usr/bin/env python
# coding: utf-8

# ## This is a shining moment for all aspiring data scientists (at least for me)
# 
# I am not the best at this kaggle competition stuff, but I join and I try. After 4 solid days of thinking about how to assign a family a day without family_id being a factor (a.k.a shuffling the family_ids), I finally wrote a function that simply assigns a day to the family. Is it perfect? Not even close. is it functional? I mean it is a function and all so I guess so. I will eventually try to find a better output with testing the giving cost functions, but it is a great day. 
# 
# Thank you to all that look at this!!

# In[ ]:


import pandas as pd
import random as rnd


# In[ ]:


data = pd.read_csv('../input/santa-workshop-tour-2019/family_data.csv')


# In[ ]:


data


# In[ ]:


d = rnd.sample(range(0,5000),5000)
day_dict = {}
fam_dict = {}
df = data
for i in range(100,0,-1):
    day_dict[i] = 0


# In[ ]:


def get_choice_for_family(df, family_id, day_dict,fam_dict):
    c=1
    row = df[df['family_id'] == family_id]
    while c < 11:
        if day_dict[row.iloc[0,c]] >289:
            c += 1
        else:
            day_dict[row.iloc[0,c]] += int(row.n_people)
            break
    fam_dict[family_id] = [row.iloc[0,c],c-1]
    return day_dict, fam_dict


# In[ ]:


for i in d:
    x = get_choice_for_family(df, i, day_dict, fam_dict)
    day_dict = x[0]
    fam_dict = x[1]


# In[ ]:


min(day_dict.values())


# In[ ]:


len(fam_dict)


# In[ ]:


fam_id = list(fam_dict.keys())


# In[ ]:


assigned_day = list(fam_dict.values())


# In[ ]:


assigned_days = [i[0] for i in assigned_day]


# In[ ]:


final = {'family_id':fam_id,'assigned_day':assigned_days}


# In[ ]:


sub = pd.DataFrame.from_dict(final)


# In[ ]:


sub = sub.sort_values(by='family_id')


# In[ ]:


sub.to_csv('sub.csv', index=False)

