#!/usr/bin/env python
# coding: utf-8

# # References
# * https://www.kaggle.com/vipito/santa-ip
# * https://www.kaggle.com/golubev/optimization-preference-cost-mincostflow
# * https://www.kaggle.com/nickel/santa-s-2019-fast-pythonic-cost-23-s
# * https://www.kaggle.com/inversion/santa-s-2019-starter-notebook

# In[ ]:


import random
import numpy as np
import pandas as pd

from numba import njit
from ortools.graph.pywrapgraph import SimpleMinCostFlow
from ortools.linear_solver.pywraplp import Solver

SEED = 1
random.seed(SEED)
np.random.seed(SEED)

N_DAYS = 100
N_FAMILIES = 5000
N_CHOICES = 10
MAX_POP = 300
MIN_POP = 125

data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')

choice_cols = [f'choice_{i}' for i in range(N_CHOICES)]
CHOICES = data[choice_cols].values - 1

COST_PER_FAMILY = [0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500]
COST_PER_MEMBER = [0,  0,  9,   9,   9,  18,  18,  36,  36, 235, 434]
F_COUNTS = data['n_people'].astype(int).values

C_COSTS = np.zeros((N_FAMILIES, N_DAYS), dtype=np.int32)
for f in range(N_FAMILIES):
    for d in range(N_DAYS):
        if d in CHOICES[f, :]:
            c = list(CHOICES[f, :]).index(d)
        else:
            c = N_CHOICES
        C_COSTS[f, d] = COST_PER_FAMILY[c] + F_COUNTS[f] * COST_PER_MEMBER[c]


# In[ ]:


@njit(fastmath=True)
def get_daily_occupancy(schedule):
    daily_occupancy = np.zeros(N_DAYS, np.int32)
    for f, d in enumerate(schedule):
        daily_occupancy[d] += F_COUNTS[f]
    return daily_occupancy


@njit(fastmath=True)
def cost_function(schedule):
    choice_cost = 0
    for f, d in enumerate(schedule):
        choice_cost += C_COSTS[f, d]
    
    daily_occupancy = get_daily_occupancy(schedule)
        
    accounting_cost = 0
    for d0 in range(N_DAYS):
        pop0 = daily_occupancy[d0]
        d1 = min(d0+1, N_DAYS-1)
        pop1 = daily_occupancy[d1]
        accounting_cost += max(0, (pop0-125.0) / 400.0 * pop0**(0.5 + abs(pop0 - pop1) / 50.0))
    
    violations = (np.count_nonzero(daily_occupancy < MIN_POP) + 
                  np.count_nonzero(daily_occupancy > MAX_POP))
    penalty = int(violations * 10e8)
    
    return choice_cost, accounting_cost, penalty


def fix_schedule(schedule):
    daily_occupancy = get_daily_occupancy(schedule)
    
    f_list = np.flip(np.argsort(F_COUNTS))
    
    while (daily_occupancy.min() < MIN_POP) or           (daily_occupancy.max() > MAX_POP):
        
        for c in range(N_CHOICES):
            for f in f_list:
                n = F_COUNTS[f]
                d_old = schedule[f]
                d_new = CHOICES[f, c]

                if (daily_occupancy[d_old] > MAX_POP) and                    ((daily_occupancy[d_new] + n) <= MAX_POP):
                    schedule[f] = d_new
                    daily_occupancy[d_new] += n
                    daily_occupancy[d_old] -= n

        for c in range(N_CHOICES):
            for f in f_list:
                n = F_COUNTS[f]
                d_old = schedule[f]
                d_new = CHOICES[f, c]

                if (daily_occupancy[d_new] < MIN_POP) and                    ((daily_occupancy[d_old] - n) >= MIN_POP):
                    schedule[f] = d_new
                    daily_occupancy[d_new] += n
                    daily_occupancy[d_old] -= n
    
    return schedule


# # Initial LP solution

# In[ ]:


model = Solver('SantaLinear', Solver.GLOP_LINEAR_PROGRAMMING)

set_f = range(N_FAMILIES)
set_d = range(N_DAYS)

x = {(f, d): model.BoolVar(f'x[{f},{d}]') for f in set_f for d in CHOICES[f, :]}
y = {(d): model.IntVar(0, MAX_POP-MIN_POP, f'y[{d}]') for d in set_d}

for f in set_f:
    model.Add(model.Sum(x[f, d] for d in set_d if (f, d) in x.keys()) == 1)

pops = [model.Sum(x[f, d] * F_COUNTS[f] for f in set_f if (f, d) in x.keys()) for d in set_d]

for d0 in set_d:
    pop0 = pops[d0]
    model.Add(pop0 >= MIN_POP)
    model.Add(pop0 <= MAX_POP)

    d1 = min(d0+1, N_DAYS-1)
    pop1 = pops[d1]
    model.Add(pop0 - pop1 <= y[d])
    model.Add(pop1 - pop0 <= y[d])
    
    model.Add(y[d] <= 30)

DELTA_WEIGHT = 500
objective = model.Sum(x[f, d] * C_COSTS[f, d] for f, d in x.keys())
objective += model.Sum(y[d] for d in set_d) * DELTA_WEIGHT

model.Minimize(objective)

model.SetTimeLimit(5 * 60 * 1000)
status = model.Solve()

if status == Solver.OPTIMAL:
    print('Found Optimal Solution')
else:
    print(f'Solver Error. Status = {status}')

schedule = np.full(N_FAMILIES, -1, dtype=np.int8)

x_vals = np.zeros((N_FAMILIES, N_DAYS))
for f, d in x.keys():
    x_vals[f, d] = x[f, d].solution_value()

for f, vals in enumerate(x_vals):
    d = np.argmax(vals)
    schedule[f] = d

score = cost_function(schedule)
print(sum(score), '|', score)

schedule = fix_schedule(schedule)
score = cost_function(schedule)
print(sum(score), '|', score)


# # Choice Search

# In[ ]:


def choice_search(schedule):
    best_score = cost_function(schedule)
    
    f_list = np.flip(np.argsort(F_COUNTS))

    for f in f_list:
        d_old = schedule[f]
        for d_new in CHOICES[f, :]:
            schedule[f] = d_new

            score = cost_function(schedule)
                
            if (sum(score) < sum(best_score)) or                (sum(score) == sum(best_score) and np.random.random() < 0.5):
                best_score = score
                d_old = d_new
            else:
                schedule[f] = d_old
    return schedule


# # Min Cost Flow

# In[ ]:


def min_cost_flow(schedule):
    MIN_FAMILY = F_COUNTS.min()
    MAX_FAMILY = F_COUNTS.max()
    
    solver = SimpleMinCostFlow()
    
    occupancy = np.zeros((N_DAYS, MAX_FAMILY+1), dtype=np.int32)
    for f, n in enumerate(F_COUNTS):
        f_node = int(f)
        f_demand = -1
        solver.SetNodeSupply(f_node, f_demand)
        
        d = schedule[f]
        occupancy[d, n] += 1
        
    for d in range(N_DAYS):
        for n in range(MIN_FAMILY, MAX_FAMILY):
            occ_node = int(N_FAMILIES + (n-2) * N_DAYS + d)
            occ_supply = int(occupancy[d, n])
            solver.SetNodeSupply(occ_node, occ_supply)

    for f, n in enumerate(F_COUNTS):
        f_node = int(f)
        
        for c in range(N_CHOICES):
            d = CHOICES[f, c]
            c_cost = int(C_COSTS[f, d])
            occ_node = int(N_FAMILIES + (n-2) * N_DAYS + d)
            solver.AddArcWithCapacityAndUnitCost(occ_node, f_node, 1, c_cost)

    status = solver.SolveMaxFlowWithMinCost()

    if status == SimpleMinCostFlow.OPTIMAL:
        for arc in range(solver.NumArcs()):
            if solver.Flow(arc) > 0:
                head = solver.Head(arc)

                if head in range(N_FAMILIES):
                    f = head
                    n = F_COUNTS[f]
                    occ_node = solver.Tail(arc)
                    d = occ_node - N_FAMILIES - (n-2) * N_DAYS
                    schedule[f] = d
    else:
        print(f'Solver Error. Status = {status}')

    return schedule


# # Swap Search

# In[ ]:


def swap_search(schedule):
    best_score = cost_function(schedule)
    
    f_list = np.random.permutation(N_FAMILIES)
    
    for f0 in f_list:
        d0 = schedule[f0]
        c0 = list(CHOICES[f0, :]).index(d0)

        swapped = False
        for d1 in CHOICES[f0, 0:c0]:
            f1_set = np.where(schedule == d1)[0]
            for f1 in f1_set:
                if d0 in CHOICES[f1, :]:
                    schedule[f0] = d1
                    schedule[f1] = d0
                    score = cost_function(schedule)
                    
                    if (sum(score) < sum(best_score)) or                        (sum(score) == sum(best_score) and np.random.random() < 0.5):
                        best_score = score
                        swapped = True
                        break
                    else:
                        schedule[f0] = d0
                        schedule[f1] = d1
            if swapped: break
                
    return schedule


# # Random Hill Climbing

# In[ ]:


def random_climb(schedule, repeats=100000):
    best_score = cost_function(schedule)

    for _ in range(repeats):
        f = np.random.randint(N_FAMILIES)
        c = np.random.randint(N_CHOICES)

        d_old = schedule[f]
        d_new = CHOICES[f, c]

        schedule[f] = d_new
        score = cost_function(schedule)

        if (sum(score) < sum(best_score)) or            (sum(score) == sum(best_score) and np.random.random() < 0.5):
            best_score = score
        else:
            schedule[f] = d_old

    return schedule


# # Ensemble

# In[ ]:


best_score = cost_function(schedule)

no_improvement = 0
while no_improvement < 5:
    improved = False
    
    while True:
        schedule = random_climb(schedule)
        score = cost_function(schedule)
        print('Random  :', sum(score), '|', score)
        if sum(score) < sum(best_score):
            best_score = score
            improved = True
        else:
            break
    
    while True:
        schedule = swap_search(schedule)
        score = cost_function(schedule)
        print('Swaps   :', sum(score), '|', score)
        if sum(score) < sum(best_score):
            best_score = score
            improved = True
        else:
            break
    
    while True:
        schedule = choice_search(schedule)
        score = cost_function(schedule)
        print('Choice  :', sum(score), '|', score)
        if sum(score) < sum(best_score):
            best_score = score
            improved = True
        else:
            break
    
    if not improved:
        schedule = min_cost_flow(schedule)
        score = cost_function(schedule)
        print('MinCost :', sum(score), '|', score)
        if sum(score) < sum(best_score):
            best_score = score
            improved = True
    
    no_improvement = 0 if improved else no_improvement + 1


# In[ ]:


submission = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv')
submission['assigned_day'] = schedule + 1
submission.to_csv('submission.csv', index=False)

