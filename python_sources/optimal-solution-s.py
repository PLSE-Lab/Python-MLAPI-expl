#!/usr/bin/env python
# coding: utf-8

# ## Referrences:
# * https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/126374 (daily occupancies)
# * https://www.kaggle.com/golubev/manual-to-improve-submissions (plots)
# * https://www.kaggle.com/nagadomi/mipcl-example-only-preference (preference cost solution)
# 
# 

#  ## Preference cost

# Minimizing preference cost is an easy to solve problem (5000x10 variables), solved here:
# https://www.kaggle.com/nagadomi/mipcl-example-only-preference 
# in about a 1 minute; discussed here: https://www.kaggle.com/mihaild/lower-bound-on-preference-cost#685643.
# As you can see on the plot below, this solution completely *ignores* occupancies between adjacent days:

# In[ ]:


from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

NUMBER_DAYS = 100
NUMBER_FAMILIES = 5000
MAX_BEST_CHOICE = 5
data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')
submission = pd.read_csv('/kaggle/input/santas-workshop-tour-2019-optimal-solution/preference_cost_43622.csv')
assigned_days = submission['assigned_day'].values
columns = data.columns[1:11]
DESIRED = data[columns].values
COST_PER_FAMILY        = [0,50,50,100,200,200,300,300,400,500]
COST_PER_FAMILY_MEMBER = [0, 0, 9,  9,  9, 18, 18, 36, 36,235]
N_PEOPLE = data['n_people'].values

def get_daily_occupancy(assigned_days):
    daily_occupancy = np.zeros(100, int)
    for fid, assigned_day in enumerate(assigned_days):
        daily_occupancy[assigned_day-1] += N_PEOPLE[fid]
    return daily_occupancy

def days_plot(assigned_days):
    daily_occupancy = get_daily_occupancy(assigned_days)
    best_choices = get_daily_occupancy(DESIRED[:,0])
    plt.rcParams['figure.figsize'] = [20, 5]
    plt.xticks(np.arange(1, 101, step=1), rotation=90)
    plt.axhline(y=125, color='gray', linestyle=':')
    plt.axhline(y=300, color='gray', linestyle=':')
    mondays125     = np.array([(day+1, daily_occupancy[day]) for day in range(100) if day % 7 == 1 and daily_occupancy[day] == 125])
    other_mondays  = np.array([(day+1, daily_occupancy[day]) for day in range(100) if day % 7 == 1 and daily_occupancy[day] != 125])
    weekends       = np.array([(day+1, daily_occupancy[day]) for day in range(100) if day % 7 in [2,3,4] or day == 0])
    not_weekends   = np.array([(day+1, daily_occupancy[day]) for day in range(1, 100) if day % 7 in [0,5,6]])
    plt.bar(*weekends.transpose()      , color = 'y', label = 'Weekends')
    plt.bar(*not_weekends.transpose()  , color = 'b', label = 'Thu-Wed-Tue')
    plt.bar(*other_mondays.transpose() , color = 'm', label = 'Mondays > 125')
    plt.bar(*mondays125.transpose()    , color = 'g', label = 'Mondays = 125')
    plt.plot(range(1,101), best_choices, color = 'k', label = 'Best choices')
    plt.ylim(0, 500)
    plt.xlim(0, 101)
    plt.xlabel('Days before Christmas', fontsize=14)
    plt.ylabel('Occupancy', fontsize=14)
    plt.legend()
    plt.show()

def days_plot2(assigned_days):
    daily_occupancy = get_daily_occupancy(assigned_days)
    best_choices = get_daily_occupancy(DESIRED[:,0])
    plt.rcParams['figure.figsize'] = [20, 5]
    plt.xticks(np.arange(1, 101, step=1), rotation=90)
    plt.axhline(y=125, color='gray', linestyle=':')
    plt.axhline(y=300, color='gray', linestyle=':')
    mondays     = np.array([(day+1, daily_occupancy[day]) for day in range(100) if day % 7 == 1])
    weekends       = np.array([(day+1, daily_occupancy[day]) for day in range(100) if day % 7 in [2,3,4] or day == 0])
    not_weekends   = np.array([(day+1, daily_occupancy[day]) for day in range(1, 100) if day % 7 in [0,5,6]])
    plt.bar(*weekends.transpose()      , color = 'y', label = 'Weekends')
    plt.bar(*not_weekends.transpose()  , color = 'b', label = 'Thu-Wed-Tue')
    plt.bar(*mondays.transpose() , color = 'm', label = 'Mondays')
    plt.plot(range(1,101), best_choices, color = 'k', label = 'Best choices')
    plt.ylim(0, 500)
    plt.xlim(0, 101)
    plt.xlabel('Days before Christmas', fontsize=14)
    plt.ylabel('Occupancy', fontsize=14)
    plt.legend()
    plt.show()    
    
def cost_function(prediction):
    N_DAYS = 100
    MAX_OCCUPANCY = 300
    MIN_OCCUPANCY = 125
    penalty = 0
    days = list(range(N_DAYS,0,-1))
    tmp = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')
    family_size_dict = tmp[['n_people']].to_dict()['n_people']

    cols = [f'choice_{i}' for i in range(10)]
    choice_dict = tmp[cols].to_dict()

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
        if  (v < MIN_OCCUPANCY): #(v > MAX_OCCUPANCY) or
            penalty += 100000000

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_costs = [max(0, accounting_cost)]
    diffs = [0]
    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_costs.append(max(0, (today_count-125.0) / 400.0 * today_count**(0.5 + diff / 50.0)))
        yesterday_count = today_count

    return penalty, sum(accounting_costs), penalty + sum(accounting_costs)
def print_cost_function(prediction):
    cfa = cost_function(prediction)
    print("Preference cost: ",cfa[0])
    print("Accounting penalty: ",cfa[1])
    print("Score: ",cfa[2])
days_plot(assigned_days)


# In[ ]:


print_cost_function(assigned_days)


# ## Accounting penalty

# I have not found any solution or discussion about minimizing only accounting penalty. So I decided to solve it for an exercise. At first glance it looks little bit more complicated than the original problem (5000x10 + 176x176x100 binary variables). On the other hand, N_d approx  N_d+1 seems to be close to optimum. After testing few scenarios, Gurobi found and prove solution presented below:

# In[ ]:


submission = pd.read_csv('/kaggle/input/santas-workshop-tour-2019-optimal-solution/accounting_penalty_313.27.csv')
assigned_days = submission['assigned_day'].values
days_plot2(assigned_days)


# In[ ]:


print_cost_function(assigned_days)


# It's interesting that daily occupancies are slowly decreasing (however there are no tricks with occupancies equal to 125!):

# In[ ]:


get_daily_occupancy(assigned_days)


# ## Score = preference cost + accounting penalty
# There are many ways of obtaining optimal solution, described in discussions (I used first one):
# * https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/126374
# * https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/126225
# * https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/126185
# 
# In general, it's an art of using good MIP solvers. I'm presenting pulp solution (python API to LP and MIP solvers) which allows using GUROBI (just in one line if you have installed and licensed client). Do not forget to setting up MIPGap = 0 :)

# In[ ]:


submission = pd.read_csv('/kaggle/input/santas-workshop-tour-2019-optimal-solution/submission_68888.04.csv')
assigned_days = submission['assigned_day'].values


# In[ ]:


get_ipython().system('pip install pulp')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfrom pulp import *\n\ndaily_occupancy = get_daily_occupancy(assigned_days)\n    \nlower_bounds = np.zeros(NUMBER_DAYS)\nupper_bounds = np.zeros(NUMBER_DAYS)\n# good seed described in discussion:\n# daily_occupancy = np.array([300, 287, 300, 300, 286, 262, 250, 250, 271, 296, 300, 300, 279, 264, 256, 273, 294, 282, 259, 228, 196, 164, 125, 300, 300, 295, 278, 263, 257, 253, 279, 276, 252, 219, 188, 156, 125, 283, 271, 248, 216, 184, 159, 125, 300, 282, 257, 226, 194, 161, 125, 286, 264, 236, 201, 168, 137, 125, 266, 241, 207, 166, 125, 125, 125, 253, 225, 190, 147, 125, 125, 125, 227, 207, 175, 129, 125, 125, 125, 235, 220, 189, 147, 125, 125, 125, 256, 234, 202, 161, 125, 125, 125, 234, 214, 181, 136, 125, 125, 125])\n# delta = 5\ndelta = 0\n\nfor fi in range(NUMBER_DAYS):\n    lower_bounds[fi] = max(daily_occupancy[fi]-delta,125)\n    upper_bounds[fi] = min(daily_occupancy[fi]+delta,300)\n        \n# Create the \'prob\' variable to contain the problem data\nprob = LpProblem("Setup occupation of days", LpMinimize)\nPCOSTM, B = {}, {} # cost matrix, boolean vars matrix\n    \nfor fid in range(NUMBER_FAMILIES):\n    for i in range(MAX_BEST_CHOICE):\n        B[fid, DESIRED[fid][i]-1] = LpVariable(f\'b{fid, i}\', 0, 1, LpInteger) # B[family, choice_day] = boolean variable\n        # setting up initial values\n        if assigned_days[fid] == DESIRED[fid][i]:\n            B[fid, DESIRED[fid][i]-1].setInitialValue(1)\n        else:\n            B[fid, DESIRED[fid][i]-1].setInitialValue(0)\n        PCOSTM[fid, DESIRED[fid][i]-1] = COST_PER_FAMILY[i] + N_PEOPLE[fid] * COST_PER_FAMILY_MEMBER[i]  \n\nD = {} # days occupancies variables matrix\nfor j in range(NUMBER_DAYS):\n    rj = range(int(lower_bounds[j]),int(upper_bounds[j])+1)\n    for i in rj:\n        if j<99:\n            rj1 = range(int(lower_bounds[j+1]),int(upper_bounds[j+1])+1)\n            for i1 in rj1:\n                D[j, i,i1] =  LpVariable(f\'D{j, i,i1}\', 0, 1, LpInteger) # day j occupancy = i and day j+1 occupancy = i1\n                # setting up initial values\n                if daily_occupancy[j] == i and daily_occupancy[j+1] == i1:\n                    D[j, i,i1].setInitialValue(1)\n                else:\n                    D[j, i,i1].setInitialValue(0)\n        else:\n            D[j,i,i] =  LpVariable(f\'D{j, i,i}\', 0, 1, LpInteger)\n            # setting up initial values\n            if daily_occupancy[j] == i:\n                D[j, i,i].setInitialValue(1)\n            else:\n                D[j, i,i].setInitialValue(0)\n\n# defining objective: preference cost + accounting penalty                \nprob += lpSum(PCOSTM[i, j] * B[i, j] for i, j in B) + lpSum(D[j,i,i1]*(int(i)-125.0)/400.0*int(i)**(0.5+abs(i-i1)/50.0) for j,i,i1 in D) \n    \nI = {}\n\nfor j in range(NUMBER_DAYS):\n    I[j] = LpVariable(f\'I{j}\', int(lower_bounds[j]), int(upper_bounds[j]), LpInteger)\n    I[j].setInitialValue(daily_occupancy[j])\n    prob += lpSum([N_PEOPLE[i] * B[i, j] for i in range(NUMBER_FAMILIES) if (i,j) in B]) == I[j]\n\nfor j in range(NUMBER_DAYS):\n    rj = range(int(lower_bounds[j]),int(upper_bounds[j])+1)\n    if j<99:\n        rj1 = range(int(lower_bounds[j+1]),int(upper_bounds[j+1])+1)\n        prob += lpSum([D[j, i,i1]*i for i in rj for i1 in rj1]) == I[j]\n        prob += lpSum([D[j, i,i1]*i1 for i in rj for i1 in rj1]) == I[j+1]\n    else:\n        prob += lpSum([D[j, i,i]*i for i in rj]) == I[j]\n\nfor i in range(NUMBER_FAMILIES):\n    prob += lpSum(B[i, j] for j in range(NUMBER_DAYS) if (i,j) in B) == 1          \n    \nprob.solve()\n\n## USING GUROBI:\n#prob.solve(GUROBI_CMD(msg =1, mip_start=1,options = [(\'MIPGap\',0),(\'SolFiles\',\'./solution/mymodel\')]))\n    \nprint("Status:", LpStatus[prob.status])\nprint("Score = ", value(prob.objective))')


# In[ ]:


days_plot(assigned_days)


# As pointed here: https://www.kaggle.com/golubev/manual-to-improve-submissions good seed-solutions have fixed some Mondays occupancy to 125. It was little bit suprising that Monday day 30 haven't occupancy 125, instead of Monday day 23!

# In[ ]:


print_cost_function(assigned_days)


# In[ ]:


submission['assigned_day'] = assigned_days
submission.to_csv('submission_68888.04.csv', index=False)


# Greetings!
