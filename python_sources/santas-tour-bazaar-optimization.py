#!/usr/bin/env python
# coding: utf-8

# # Welcome to Santa's Workshop Tour Bazaar 2019
# This is a starter code and easy to understand. 
# 
# We consider a linear optimization problem to minimize a cost function subject to constraints. The cost function contains the family (preference) and accounding (penalty) costs. The constraints are: For each day the total number of people attending the workshop must be between 125 and 300. 
# 
# We use code snipes from the starter notebook: https://www.kaggle.com/inversion/santa-s-2019-starter-notebook
# 
# The algorithm: We create a simple feasible start solution for the first iteration step. For this we say that every day get the same number fo families. This number is 50. After that the families go to the bazaar to reduce the total costs. Each familie with preference costs greater than 0 tries to swap the day with another family or swap the assigned day. But the swap is only valid if it changes the preference or acoounding costs.
# 
# The convergence of the algorithm: By using the GPU we can realize 3 iteration steps to get total costs of 288.002. If you are able to do 10 iterations steps (not on kaggle for this algorithm) you will see the following results:
# 
# |iteration step| objective value|
# |---------------|-------------|
#  | 0 | 10.645.224 |   
#  | 1 | 719.769 |   
#  | 2 | 363.117 |   
#  | 3 | 288.002 |   
#  | 4 | 259.360 |   
#  | 5 | 249.604 |   
#  | 6 | 246.836 |   
#  | 7 | 245.979 |   
#  | 8 | 244.238 |   
#  | 9 | 243.764 |   
#  | 10 | 243.317 | 
#  
#  We can see that the improvement of the objective value is small in the later steps. Alternatively you can also start with another start solution which is closer to the optimum. 
#  
# For this version we start with a precalculated start solution. 

# # Load Libraries

# In[ ]:


import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import os


# # Input Data

# In[ ]:


path_in = '../input/santa-workshop-tour-2019/'
path_start_solution = '../input/santa-workshop-tour-start-solution/'
print('input_data:', os.listdir(path_in))
print('start solution:', os.listdir(path_start_solution))


# # Read Data
# Here you can also load a start solution.

# In[ ]:


data = pd.read_csv(path_in+'family_data.csv')
data.index = data['family_id']
samp_subm = pd.read_csv(path_in+'sample_submission.csv')
start_solution = pd.read_csv(path_start_solution+'santa_workshop_tour_start_solution_01.csv', index_col=0)


# # Parameter

# In[ ]:


num_days = 100
lower = 125
upper = 300
days = list(range(num_days, 0, -1))


# # Functions
# We define some function for the method used below. The cost functions based on the starter code: https://www.kaggle.com/inversion/santa-s-2019-starter-notebook
# ## Calc Family Costs

# In[ ]:


def calc_family_costs(family):
    assigned_day = family['assigned_day']
    number_member = family['n_people']
    if assigned_day == family['choice_0']:
        penalty = 0
    elif assigned_day == family['choice_1']:
        penalty = 50
    elif assigned_day == family['choice_2']:
        penalty = 50 + 9 * number_member
    elif assigned_day == family['choice_3']:
        penalty = 100 + 9 * number_member
    elif assigned_day == family['choice_4']:
        penalty = 200 + 9 * number_member
    elif assigned_day == family['choice_5']:
        penalty = 200 + 18 * number_member
    elif assigned_day == family['choice_6']:
        penalty = 300 + 18 * number_member
    elif assigned_day == family['choice_7']:
        penalty = 300 + 36 * number_member
    elif assigned_day == family['choice_8']:
        penalty = 400 + 36 * number_member
    elif assigned_day == family['choice_9']:
        penalty = 500 + 36 * number_member + 199 * number_member
    else:
        penalty = 500 + 36 * number_member + 398 * number_member
    return penalty


# ## Accounting Costs

# In[ ]:


def calc_accounting_cost(data):
    accounting_cost = 0
    daily_occupancy = {k:0 for k in days}
    family_size_dict = data[['n_people']].to_dict()['n_people']
    for f, d in enumerate(data['assigned_day']):
        n = family_size_dict[f]
        daily_occupancy[d] += n

    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)
    accounting_cost = max(0, accounting_cost)
    
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
        yesterday_count = today_count
    return accounting_cost


# ## Plot Function
# We use a plot function to visualize the total costs of each iteration step and check the convergence.

# In[ ]:


def plot_results(data, name):
    x = data.columns
    y = data.loc[name]
    plt.plot(x, y, 'ro')
    plt.grid()
    plt.title(name)
    plt.xlabel('steps')
    plt.ylabel('value')
    plt.show()


# ## Check day
# To check a swap we have to test the constraint explained before.

# In[ ]:


def check_day(data, day):
    group_data = data.groupby('assigned_day').sum()['n_people'].to_frame()
    if (125 <= group_data.loc[day, 'n_people']) & (group_data.loc[day, 'n_people'] <= 300):
        return True
    else:
        return False


# # Create A Start Solution
# We use a simple distribution and test the above functions to calc the cost per family.

# In[ ]:


for i in range(num_days):
    data.loc[i*50:(i+1)*50-1, 'assigned_day'] = i+1
data['assigned_day'] = data['assigned_day'].astype(int)


# Load a start solution.

# In[ ]:


data['assigned_day'] = start_solution['assigned_day']


# Have a look on the data.

# In[ ]:


data.head()


# Calc the family costs for a given family_id.

# In[ ]:


family_id = 100
calc_family_costs(data.iloc[family_id])


# Calc the cost for all families.

# In[ ]:


data['penalty_cost'] = data.apply(calc_family_costs, axis=1)


# In[ ]:


data.head()


# Calc the accounting costs for all days and all families.

# In[ ]:


acc_costs = calc_accounting_cost(data)
acc_costs


# Total costs for the start solution are the sum over the family costs and the accounting cost.

# In[ ]:


print('Total costs:', data['penalty_cost'].sum()+ acc_costs)


# # Method To Reduce The Total Costs
# The main idea is to change the day with another family or swap the assigned day with respect to the constraints.

# In[ ]:


def check_swap_day(data, family, choice):
    data_copy = data.copy()
    data_copy.loc[family, 'assigned_day'] = data_copy.loc[family, 'choice_'+str(choice)]
    data_copy.loc[family, 'penalty_cost'] = calc_family_costs(data_copy.iloc[family])
    
    penalty_before = data.loc[family, 'penalty_cost']
    accounting_before = calc_accounting_cost(data)
    
    penalty_after = data_copy.loc[family, 'penalty_cost']
    accounting_after = calc_accounting_cost(data_copy)
    
    # Check conditions
    day_before = check_day(data_copy, data.loc[family, 'assigned_day'])
    day_after = check_day(data_copy, data_copy.loc[family, 'assigned_day'])

    if(day_before==True and day_after==True):
        improvement = (penalty_before-penalty_after)+(accounting_before-accounting_after)
    else:
        improvement = -1
    
    return improvement


# Test the check_swap_day function.

# In[ ]:


family_id = 386
check_swap_day(data, family_id, 0)


# In[ ]:


def check_swap_family(data, family, choice):
    family1 = family
    day_family1 = data.loc[family1, 'assigned_day']
    penalty1 = data.loc[family1, 'penalty_cost']
    member_family1 = data.loc[family1, 'n_people']
    
    day_member_list = data.groupby('assigned_day')['family_id'].apply(list).to_frame()
    
    improvements = {}
    for member in day_member_list.loc[data.loc[family1, 'choice_'+str(choice)], 'family_id']:
        family2 = member
        day_family2 = data.loc[family2, 'assigned_day']
        member_family2 = data.loc[family2, 'n_people']
        penalty2 = data.loc[family2, 'penalty_cost']
        
        # simulate the swap with another family
        data_copy = data.copy()
        data_copy.loc[family2, 'assigned_day'] = data_copy.loc[family1, 'assigned_day']
        data_copy.loc[family1, 'assigned_day'] = data_copy.loc[family1, 'choice_'+str(choice)]
        # calc the new penalty cost for both families
        new_penalty1 = calc_family_costs(data_copy.iloc[family1])
        new_penalty2 = calc_family_costs(data_copy.iloc[family2])
        # check both days before and after swaping
        day_before = check_day(data_copy, data.loc[family1, 'assigned_day'])
        day_after = check_day(data_copy, data_copy.loc[family1, 'choice_'+str(choice)])
        # calc the accounting costs before and after swaping
        accounting_before = calc_accounting_cost(data)
        accounting_after = calc_accounting_cost(data_copy)
        if(day_before==True and day_after==True):
            improvement = (penalty1-new_penalty1) + (penalty2-new_penalty2) + (accounting_before-accounting_after)
        else:
            improvement = -1
        improvements.update({member:improvement})
   
    maximum = max(zip(improvements.values(), improvements.keys()))
    family_swap = maximum[1]
    return improvement, family_swap


# Test the check_swap_family function.

# In[ ]:


family_id = 386
check_swap_family(data, family_id, 0)


# Compare the check_swap_day and check_swap_family functions.

# In[ ]:


family_id = 386
choice = 0
improvement_day = check_swap_day(data, family_id, choice)
improvement_family, family_swap = check_swap_family(data, family_id, choice)
improvement_day, improvement_family, family_swap


# Combine the check_swap_day and check_swap_family functions to the algorithm.

# In[ ]:


def go_to_bazaar(data, family):
    family1 = family
    day_family1 = data.loc[family1, 'assigned_day']
    penalty1 = data.loc[family1, 'penalty_cost']
    member_family1 = data.loc[family1, 'n_people']
    
    status = False
    
    for choice in range(10):
        """ Should i swap the day? """
        improvement_day = check_swap_day(data, family1, choice)
        """ Should i swap with another family? """
        improvement_family, family2 = check_swap_family(data, family1, choice)
    
        if(improvement_day >= 0 or improvement_family >= 0):
            if(improvement_day > improvement_family):
                #print('swap day')
                data.loc[family, 'assigned_day'] = data.loc[family, 'choice_'+str(choice)]
                data.loc[family, 'penalty_cost'] = calc_family_costs(data.iloc[family])
                status = True
            else:
                #print('swap family')
                data.loc[family2, 'assigned_day'] = data.loc[family1, 'assigned_day']
                data.loc[family1, 'assigned_day'] = data.loc[family1, 'choice_'+str(choice)]
        
                data.loc[family1, 'penalty_cost'] = calc_family_costs(data.iloc[family1])
                data.loc[family2, 'penalty_cost'] = calc_family_costs(data.iloc[family2])
                status = True
            if(status==True):
                break


# Test the go_to_bazaar function.

# In[ ]:


family_id = 386
#go_to_bazaar(data, family_id)
#print('Total costs:', data['penalty_cost'].sum(), calc_accounting_cost(data))


# # Store Results
# We want to analyse the results of the iteration steps.

# In[ ]:


results = pd.DataFrame()
results[0] = data['penalty_cost'].describe()
results.loc['costs', 0] = data['penalty_cost'].sum()+calc_accounting_cost(data)


# # Iterations

# In[ ]:


num_steps = 10

for step in range(num_steps):
    print('step: ', step)
    families_high_scored = list(data[data['penalty_cost']>0].index)
    print('# families: ', len(families_high_scored),
          'first:', families_high_scored[0],
          'last:', families_high_scored[-1])
    for family in families_high_scored:
        #print('   family:', family)
        go_to_bazaar(data, family)
    data['penalty_cost'] = data.apply(calc_family_costs, axis=1)
    print('costs:', data['penalty_cost'].sum(), calc_accounting_cost(data))
    results[step+1] = data['penalty_cost'].describe()
    results.loc['costs', step+1] = data['penalty_cost'].sum()+calc_accounting_cost(data)


# # Analyse Results

# In[ ]:


results = results.reindex(sorted(results.columns), axis=1)


# In[ ]:


plot_results(results, 'costs')


# In[ ]:


plot_results(results, 'mean')


# # Final costs

# In[ ]:


print('Total costs:', data['penalty_cost'].sum() + calc_accounting_cost(data))


# # Write Output

# In[ ]:


data = data.sort_index()
output = pd.DataFrame({'family_id': samp_subm.index,
                       'assigned_day': data['assigned_day']})
output.to_csv('submission.csv', index=False)


# In[ ]:


import pandas as pd
santa_workshop_tour_start_solution_01 = pd.read_csv("../input/santa-workshop-tour-start-solution/santa_workshop_tour_start_solution_01.csv")

