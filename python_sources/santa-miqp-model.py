#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ortools.linear_solver import pywraplp

Data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')


# In[ ]:


F = range(5000)
D = range(1,101)
P = range(125,301)

# family choice:
FC = Data.loc[:, 'choice_0': 'choice_9'].values

# family size:
FS = Data.loc[:,'n_people'].values

# preference cost:
PC = {(f, d):
      0 if d == FC[f,0] else
     50 if d == FC[f,1] else
     50  + FS[f]*9 if d == FC[f,2] else
     100 + FS[f]*9 if d == FC[f,3] else
     200 + FS[f]*9 if d == FC[f,4] else
     200 + FS[f]*18 if d == FC[f,5] else
     300 + FS[f]*18 if d == FC[f,6] else
     300 + FS[f]*36 if d == FC[f,7] else
     400 + FS[f]*36 if d == FC[f,8] else
     500 + FS[f]*235 if d == FC[f,9] else
     500 + FS[f]*434 for f in F for d in D}

# accounting penalty:
AP = {(i, j): (i-125)/400 * i**(1/2+abs(i-j)/50) for i in P for j in P}

# accounting penalty for d=100:
AP100 = {i: (i-125)/400 * i**(1/2) for i in P}


# In[ ]:


#Deer = pywraplp.Solver('SantaCBCDeer', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
Deer = pywraplp.Solver('SantaCLPDeer', pywraplp.Solver.CLP_LINEAR_PROGRAMMING)
#Deer = pywraplp.Solver('SantaGLOPDeer', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
#Deer = pywraplp.Solver('SantaBOPDeer', pywraplp.Solver.BOP_INTEGER_PROGRAMMING)

Deer.SetNumThreads(4)
Deer.set_time_limit(300*1000)
Deer.EnableOutput()

X = {(f, d): Deer.BoolVar(f'X[{f},{d}]') for f in F for d in D}
        
for f in F:
    Deer.Add(Deer.Sum(X[f,d] for d in D) == 1)

N = {d: Deer.Sum(X[f,d] * FS[f] for f in F) for d in D}
    
Y = {(p, d): Deer.BoolVar(f'Y[{p},{d}]') for p in P for d in D}
        
for d in D:
    Deer.Add(N[d] >= 125)
    Deer.Add(N[d] <= 300)
    Deer.Add(Deer.Sum(Y[p,d] for p in P) == 1)
    Deer.Add(Deer.Sum(Y[p,d] * p for p in P) == N[d])
    
pref_cost = Deer.Sum(X[f,d] * PC[f,d] for f in F for d in D)

#acc_penalty_days_1_99 = Deer.Sum(Y[i,d] * Y[j,d+1] * AP[i,j] for d in range(1,100) for i in P for j in P)

#acc_penalty_day_100 = Deer.Sum(Y[i,100] * AP100[i] for i in P)

score = pref_cost# + acc_penalty_day_100# + acc_penalty_days_1_99
                        
Deer.Minimize(score)

status = Deer.Solve()
    
DeerStatus = {0:'OPTIMAL', 1:'FEASIBLE', 2:'INFEASIBLE',
              3:'UNBOUNDED', 4:'ABNORMAL', 5:'MODEL_INVALID', 6:'NOT_SOLVED'}
    
print('Deer model result:', DeerStatus[status])

if status == Deer.OPTIMAL or status == Deer.FEASIBLE:
    print('Deer objective value =', Deer.Objective().Value())

