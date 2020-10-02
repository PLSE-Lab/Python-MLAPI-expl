#!/usr/bin/env python
# coding: utf-8

# The goal is this notebook is to provide a few "do it yourself" graph theoretic algorithms.
# A, Moving 1 family. 
# B, Exchanging several families of the same sizes. 
# C, Moving multiple families of different sizes.
# From the sample submission it achives less than 73000. 
# With better initialization one can greatly improve (My best score was 69530) 

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


# ## Read in the family information and sample submission

# In[ ]:


newfile ='/kaggle/input/santa-workshop-tour-2019/sample_submission.csv'


# In[ ]:


def FindLoop(Edges, Vertices, n):
    dist = {}
    Source = {}
    for a in Vertices:
        for t in range(n+1):
            Source[(a,t)] = a # moving from Source to a at time t
    for a in Vertices:
        for t in range(n+2):
            dist[(a,t)] = 0 # minimal distance to a from root in t steps or less
    
    for i in range(n+1):
        for a in Vertices:
            dist[(a,i+1)] = dist[(a,i)]
        for (a, b, w) in Edges:
                if dist[(b,i+1)] > dist[(a,i)]+w:
                    dist[(b,i+1)] = dist[(a,i)]+w
                    Source[(b,i+1)] = a
    Vert = []
    for b in Vertices:
        if dist[(b,n+1)] < dist[(b,n)] - 0.1:
            Vert.append(b)
    if Vert == []:
        return []
    
    Loops = []
    for B in Vert:
        List = [B]
        for i in range(n+1,0, -1):
            b = List[-1]
            a = Source[(b,i)]
            if a != b:
                List.append(a)
        End = -1
        for i in range(1,len(List)):
            if List[i] in List[:i]:
                End = i
                break
        if End == -1:
            continue
        for i in range(len(List)):
            if List[i] == List[End]:
                Start = i
                break
        FinalList = List[Start: End+1]
        FinalList.reverse()
        Loops.append(FinalList)
    return Loops  


# In[ ]:


Old = pd.read_csv(newfile)


# In[ ]:


fpath = '/kaggle/input/santa-workshop-tour-2019/family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

fpath = '/kaggle/input/santa-workshop-tour-2019/sample_submission.csv'
submission = pd.read_csv(fpath, index_col='family_id')


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


GoodDays = {}
for i in range(5000):
    ans = [choice_dict['choice_'+str(j)][i] for j in range(10)]
    GoodDays[i] = ans

    


# In[ ]:


def cal_cost(n):
    arr = np.zeros((11,))
    arr[0] = 0
    arr[1] = 50
    arr[2] = 50 + 9 * n
    arr[3] = 100 + 9 * n
    arr[4] = 200 + 9 * n
    arr[5] = 200 + 18 * n
    arr[6] = 300 + 18 * n
    arr[7] = 300 + 36 * n
    arr[8] = 400 + 36 * n
    arr[9] = 500 + 235 * n
    arr[10] = 500 + 434 * n
    return arr


# In[ ]:


cols = [f'choice_{i}' for i in range(10)]
choice = np.array(data[cols])


# In[ ]:


cost = np.zeros((5000,101),dtype='int32')
for i in range(5000):
    for j in range(101):
        cost[i,j]=cal_cost(family_size_dict[i])[10]
for i in range(5000):
    for j in range(10):
        c = choice[i,j]
        n = family_size_dict[i]
        cost[i,c] = cal_cost(n)[j]

print(cost.shape)


# In[ ]:


def cost_function(prediction):

    penalty = 0

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = {}
    for k in days:
        daily_occupancy[k] = 0
  
    
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
            penalty += 300000000

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


cost_function(submission.assigned_day)


# In[ ]:


def f(a,b):
    if a< 125 and a > 0 :
        return 300000000
    if b <125 and b >0:
        return 300000000
    if a > 300:
        return 300000000
    if b >300:
        return 300000000
    if a == 0:
        return 0
    else :
        return ((a-125)/400)*(a**(0.5+ (abs(a-b)/50)))


# In[ ]:


best = Old['assigned_day'].tolist()
start_score = cost_function(best)
new = best.copy()


# In[ ]:


start_score


# In[ ]:


V = [[],[],[],[],[],[],[],[],[]]
for i in range(5000):
        b = data['n_people'][i]
        V[b].append(i)


# In[ ]:


best_score = start_score


# In[ ]:


def MakeGraph1(new, k):
    R = range(1,101)
    F = {}
    H = {}
    for i in R:
        for j in R:
            F[(i,j)] = 1000000000
    for i in range(5000):
        day1 = new[i]
        n = data['n_people'][i]
        if n != k:
            continue
        else :
            for day2 in R:
                C = cost[i,day2]-cost[i,day1]
                if  C < F[(day1,day2)]:
                    F[(day1,day2)] = C
                    H[(day1,day2)] = i
   
    V = list(R)
    E = []
    for day1 in R:
        for day2 in R:
            if day1 == day2:
                continue
            else :
                E.append((day1,day2,F[(day1,day2)]))
                
    
    
    x = FindLoop(E,V,len(V))
    if x != []:
        x = x[0]
        for i in range(len(x)-1):
            day1 = x[i]
            day2 = x[i+1]
            family = H[(day1,day2)]
            new[family]= day2
    
    return new


# In[ ]:


def MakeGraph2(new,parity, m, M):
    
    Daily = np.zeros(102)
    if parity == -1:
        R = range(1,101)
    if parity == 0:
        R = range(2,102,2)
    if parity == 1:
        R = range(1,101,2)
    if parity == 2:
        R = list(range(1,51,2)) + list(range(52,102,2))
    if parity == 3:
        R = list(range(2,50,2)) + list(range(51,101,2))
    
    A1 = list(range(1,25,2))
    A2 = list(range(2,25,2))
    B1 = list(range(26,50,2))
    B2 = list(range(27,50,2))
    C1 = list(range(51,75,2))
    C2 = list(range(52,75,2))
    D1 = list(range(76,101,2))
    D2 = list(range(77,101,2))
    if parity == 4:
        R = A1+B2+C1+D2
    if parity == 5:
        R = A1+B2+C2+D1
    if parity == 6:
        R = A2+B1+C1+D2
    if parity == 7:
        R = A2+B1+C2+D1
  
    for i in range(5000):
        a = new[i]
        b = data['n_people'][i]
        Daily[a]+=b
    Daily[101]=Daily[100]
    
    
    F = {}
    H = {}
    
    F = {}
    for day1 in R:
        for day2 in R:
            for s in range(2,9):
                F[(day1,day2,s)] = 100000000
                H[(day1,day2,s)] = []
    for i in range(5000):
        day1 = new[i]
        if day1 not in R:
            continue
        else: 
            n = data['n_people'][i]
            for day2 in R:
                C = cost[i,day2]-cost[i,day1]
                if  C < F[(day1,day2,n)]:
                    F[(day1,day2,n)] = C
                    H[(day1,day2,n)] = [i]
                    
    for day1 in R:
        for day2 in R:
            x8 = (day1,day2,8)
            x7 = (day1,day2,7)
            x6 = (day1,day2,6)
            x5 = (day1,day2,5)
            x4 = (day1,day2,4)
            x3 = (day1,day2,3)
            x2 = (day1,day2,2)
            if F[x2]+F[x6] < F[x8]:
                F[x8] = F[x2]+F[x6]
                H[x8] = H[x2]+H[x6]
            if F[x3]+F[x5] < F[x8]:
                F[x8] = F[x3]+F[x5]
                H[x8] = H[x3]+H[x5]
            if F[x2]+F[x5]< F[x7]:
                F[x7] = F[x2]+F[x5]
                H[x7] = H[x2]+H[x5]
            if F[x3]+F[x4] < F[x7]:
                F[x7] = F[x3]+F[x4]
                H[x7] = H[x3] +H[x4]
            if F[x2]+F[x4] < F[x6]:
                F[x6] = F[x2]+F[x4]
                H[x6] = H[x2]+H[x4]
            if F[x2]+F[x3] < F[x5]:
                F[x5] = F[x2]+F[x3]
                H[x5] = H[x2]+H[x3]
                
                
            
            
    
    G = {}
    for day1 in R:
        for diff in range(-6,7):
            x = Daily[day1-1]
            y = Daily[day1]
            z = Daily[day1+1]
            
            G[(day1,diff)] = f(x, y+diff)+f(y+diff,z)- f(x,y)-f(y,z)
                    
    V = []  
    for day in R:
        for size in range(m,M+1):
            V.append((day,size))
    
   
    E = []
    for day1 in R:
        for day2 in R:
            if day1 == day2:
                continue 
            for s1 in range(m,M+1):
                for s2 in range(m,M+1):
                    v1 = (day1,s1)
                    v2 = (day2,s2)
                    c = F[(day1,day2,s1)] + G[(day2, s1-s2)]
                    edge = (v1,v2, c)
                    E.append(edge)
                
    
    
    X = FindLoop(E,V,len(V))
    Best_copy = new.copy()
    Best = cost_function(new)
    for x in X:
        new2 = new.copy()
        for i in range(len(x)-1):
            day1, s1 = x[i]
            day2, s2  = x[i+1]
            Family = H[(day1,day2, s1)]
            for family in Family:
                new2[family]= day2 
        if cost_function(new2) < Best:
            Best = cost_function(new2)
            Best_copy = new2
        
    
    return Best_copy


# In[ ]:


def Loop0(new):
    Daily = np.zeros(102)
    R = range(1,101)
  
    for i in range(5000):
        a = new[i]
        b = data['n_people'][i]
        Daily[a]+=b
    Daily[101]=Daily[100]
   
    F = {}
    for day1 in R:
        for day2 in R:
            for s in range(2,9):
                F[(day1,day2,s)] = 100000000
    for i in range(5000):
        day1 = new[i]
        if day1 not in R:
            continue
        else: 
            n = data['n_people'][i]
            for day2 in R:
                F[(day1,day2,n)] = min(F[(day1,day2,n)], cost[i,day2]-cost[i,day1])
    G = {}
    for day1 in R:
        for diff in range(-6,7):
            x = Daily[day1-1]
            y = Daily[day1]
            z = Daily[day1+1]
            G[(day1,diff)] = f(x, y+diff)+f(y+diff,z)- f(x,y)-f(y,z)
            
    Best = 0
    Solution = (-1,-1, -1,-1)
    for i in R:
        for j in R:
            if  abs(i-j) <= 1 :
                continue
           
            for a in range(2,9):
                for b in range(2,9):
                    C = F[(i,j,a)]+ F[(j,i,b)] 
                    if a == b:
                        P = 0
                    else :
                        P = G[(j,a-b)]+G[(i, b-a)]
                    if P+C < Best:
                        Best = P+C
                        Solution = (i,j,a,b)
    if Best < -0.1:
        print(Best)
        a,b,d,e  = Solution
        p = -1
        q = -1
        
        
        for i in range(5000):
            if new[i] == a and data['n_people'][i] == d and cost[i,b]-cost[i,a] == F[(a,b,d)]:
                p = i
          
        for i in range(5000):
            if new[i] == b and data['n_people'][i] == e and cost[i,a]-cost[i,b] == F[(b,a,e)]:
                q = i 
    
           
       
                
        if  p == -1 or q == -1:
            print('Problem')
            
        new[p] = b
        new[q] = a
       
                
    return new
   


# In[ ]:


def Exchange(new):
    IM = 0

   
        
        
    t = 0
    for k in range(2,9):
        for i in V[k]:
            for j in V[k]:
                t+=1
                if t%1000000 == 1:
                    start_score = cost_function(new)
                    print(t, start_score)
                    
                day1 = new[i]
                day2 = new[j]
                if cost[i,day1]+cost[j,day2] > cost[i,day2]+cost[j,day1]:
                    new[i] = day2
                    new[j] = day1
                    IM+=1
                    
                elif (cost[i,day1]+cost[j,day2] == cost[i,day2]+cost[j,day1]) and np.random.randint(7) in [0,1,2,3]:
                    new[i] = day2
                    new[j] = day1
    print(IM)
    return new


# In[ ]:


def Move(new):
    IM = 0
    Daily = np.zeros(102)
  
    for i in range(5000):
        a = new[i]
        b = data['n_people'][i]
        Daily[a]+=b
    Daily[101] = Daily[100]
        
    print(Daily)
 
   
    
    
    t = 0
    fam = list(range(5000))
 
    
    
    for fam_id in fam:
        t+=1
        if t%1000 == 1:
            start_score = cost_function(new)
            print(t, start_score)
  
        for pick in range(1,101):
            new_day = pick
            
            old_day = new[fam_id]
            if new_day == old_day:
                continue
            a = data['n_people'][fam_id]
            C1 = cost[fam_id, old_day]
            C2 = cost[fam_id, new_day]
            
            if abs(old_day-new_day) > 1: 
                x1 = Daily[old_day -1]
                x2 = Daily[old_day]
                x3 = Daily[old_day+ 1]
                y1 = Daily[new_day -1]
                y2 = Daily[new_day]
                y3 = Daily[new_day+1]
                P1 = f(x1,x2)+f(x2,x3)+ f(y1,y2)+f(y2,y3)
                P2 = f(x1,x2-a)+f(x2-a,x3) + f(y1,y2+a)+f(y2+a,y3)
            
            elif new_day == old_day+1:
                x1 = Daily[old_day -1]
                x2 = Daily[old_day]
                x3 = Daily[old_day + 1]
                x4 = Daily[old_day+2]
                P1 = f(x1,x2)+f(x2,x3)+f(x3,x4)
                P2 = f(x1,x2-a)+ f(x2-a,x3+a)+f(x3+a,x4)
            
            elif new_day == old_day-1:
                x1 = Daily[old_day -2]
                x2 = Daily[old_day -1]
                x3 = Daily[old_day ]
                x4 = Daily[old_day+1]
                P1 = f(x1,x2)+f(x2,x3)+f(x3,x4)
                P2 = f(x1,x2+a)+ f(x2+a,x3-a)+f(x3-a,x4)  
                
                
            if P1+C1 > P2+C2:
                new[fam_id] = new_day
          
                Daily[old_day]-=a
                Daily[new_day]+=a
                Daily[101] = Daily[100]
                IM+=1
    print(IM)       
    return new, IM


# In[ ]:


def Loop3(new):
    Daily = np.zeros(102)
    R = range(1,101)
  
    for i in range(5000):
        a = new[i]
        b = data['n_people'][i]
        Daily[a]+=b
    Daily[101]=Daily[100]
  
    F = {}
    for day1 in R:
        for day2 in R:
            for s in range(2,9):
                F[(day1,day2,s)] = 100000000
    for i in range(5000):
        day1 = new[i]
        if day1 not in R:
            continue
        else: 
            n = data['n_people'][i]
            for day2 in R:
                F[(day1,day2,n)] = min(F[(day1,day2,n)], cost[i,day2]-cost[i,day1])
    G = {}
    for day1 in R:
        for diff in range(-6,7):
            x = Daily[day1-1]
            y = Daily[day1]
            z = Daily[day1+1]
            G[(day1,diff)] = f(x, y+diff)+f(y+diff,z)- f(x,y)-f(y,z)
            
    Best = 0
    Solution = (-1,-1,-1,-1,-1,-1)
    for i in R:
        for j in R:
            if abs(i-j) <=1 :
                continue
            if j < i:
                continue
            for k in R:
                if abs(k-i) <=1 or abs(k-j) <=1:
                    continue
                if k < i:
                    continue
                for a in range(2,9):
                    for b in range(2,9):
                        for c in range(2,9):
                            C = F[(i,j,a)]+ F[(j,k,b)] + F[(k,i,c)]
                            P = G[(j,a-b)]+G[(k, b-c)]+ G[(i,c-a )]
                            if P+C < Best:
                                Best = P+C
                                Solution = (i,j,k,a,b,c)
                                
    if Best < -0.1:
        print(Best)
        a,b,c,d,e,f1 = Solution
        p = -1
        q = -1
        r = -1
        
        for i in range(5000):
            if new[i] == a and data['n_people'][i] == d and cost[i,b]-cost[i,a] == F[(a,b,d)]:
                p = i
          
        for i in range(5000):
            if new[i] == b and data['n_people'][i] == e and cost[i,c]-cost[i,b] == F[(b,c,e)]:
                q = i 
    
           
        for i in range(5000):
            if new[i] == c and data['n_people'][i] == f1 and cost[i,a]-cost[i,c] == F[(c,a,f1)]:
                r = i  
                
        if r == -1 or p == -1 or q == -1:
            print('Problem')
            
        new[p] = b
        new[q] = c
        new[r] = a
                
    return new


# In[ ]:


#Local heuristic changes
for i in range(20):
    print('Epoch: '+ str(i) )
    new = Exchange(new)
    new = Loop0(new)
    for p in range(0,8):
        new = MakeGraph2(new, p, 2, 8)
        print(cost_function(new))
  
    for S in range(2,9):
            new = MakeGraph1(new, S)
            print(cost_function(new))

    new, _ = Move(new)
   


# In[ ]:


# Additional (and more time consuming) local changes
for i in range(8):
    print('Epoch: '+ str(i) )
    new = Exchange(new)
    new = Loop0(new)
    for p in range(-1,8):
        new = MakeGraph2(new, p, 2, 8)
        print(cost_function(new))
  
    for S in range(2,9):
            new = MakeGraph1(new, S)
            print(cost_function(new))
    
    new = Loop3(new)
    new, _ = Move(new)
   


# In[ ]:


submission['assigned_day'] = new
score = cost_function(new)
submission.to_csv(f'submission_{score}.csv')
print(f'Score: {score}')

