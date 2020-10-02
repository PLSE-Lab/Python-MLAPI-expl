#!/usr/bin/env python
# coding: utf-8

# # Does it pay to move interstate for work?
# 
# Do people who've moved somewhere
#  other than their state of birth earn more than people who stay in the same place? We'll see that the answer
# is a definite yes.
# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn

# load data
household = pd.concat([pd.read_csv('../input/pums/ss13husa.csv',usecols=['SERIALNO','ST']),pd.read_csv('../input/pums/ss13husb.csv',usecols=['SERIALNO','ST'])],axis=0)
person = pd.concat([pd.read_csv('../input/pums/ss13pusa.csv',usecols=['SERIALNO','POBP','SOCP','PERNP']),pd.read_csv('../input/pums/ss13pusb.csv',usecols=['SERIALNO','POBP','SOCP','PERNP'])],axis=0)

# merge household and person
all_data =  household.merge(person,how='inner',on='SERIALNO')

# place of birth: POBP
# place of current residence: ST
# restrict to people born in US
all_data = all_data[all_data['POBP'] <= 56]
all_data['NATIVE'] = (all_data['POBP'] == all_data['ST']).astype(int)  # 'NATIVE' = 1 when the person resides in the state they were born

# calculate income for native and transfers
avg_income_by_native_split = all_data.groupby('NATIVE').mean()['PERNP']  # income by group
moving_bonus_avg = avg_income_by_native_split[0]/avg_income_by_native_split[1]
print("People who move interstate earn {:.2f} times as much as people who stay put.".format(moving_bonus_avg))


# ## Does it depend on the job?
# Now we look at how this varies by occupation. We'll see that for people earning less than around $50,000,
# moving interstate doesn't make much difference (and in fact they may even earn less). 
# But for higher paying jobs, moving seems to pay off.

# In[ ]:



# break down by occupation
avg_income_by_occupation_and_native = all_data.groupby(['NATIVE','SOCP']).mean()['PERNP']  # income by group, occupation
avg_income_by_occupation_and_native = avg_income_by_occupation_and_native.unstack(level=0)

# moving bonus
moving_bonus_by_occupation = avg_income_by_occupation_and_native.loc[:,0]/avg_income_by_occupation_and_native.loc[:,1]
avg_income_by_occupation = all_data.groupby('SOCP').mean()['PERNP']  # income by occupation

# plot moving bonus by occupation
plt.figure()
plt.scatter(avg_income_by_occupation,moving_bonus_by_occupation)
plt.xlabel('Income')
plt.ylabel('Multiplier for moving interstate')

