#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt
import seaborn as sns


# ### Data

# In[ ]:


'''data = pd.read_csv("../input/Kerkar Initial Hard Model (UlterraTeratek).csv")
data.head(3)'''
'''data = pd.read_csv("../input/snl2015swg/SNL 2015 SWG data.csv")
data.head(20)'''
data = pd.read_csv("../input/data222/SNL 2015 SWG data2.csv")
data.head(20)


# ### Define Functions and Models

# In[ ]:


def Plot_data_seaborn():
    sns.set()
    columns = ['UCS (psi)','WOB (lbf)','RPM','Db (inch)','BR','SR','Blade Count','ROP (ft/hr)']
    sns.pairplot(data[columns], size = 1.5 , kind ='scatter')
    plt.show()
    
def Turn_data_to_seperate_lists():
    UCS = data['UCS (psi)']
    WOB = data['WOB (lbf)']
    RPM = data['RPM']
    Db = data['Db (inch)']
    BR = data['BR']
    SR = data['SR']
    Nb = data['Blade Count']
    ROP_Data  = data['ROP (ft/hr)']
    
    return UCS, WOB, RPM, Db, BR, SR, Nb, ROP_Data
    

def Kerkar_Model(UCS, WOB, RPM, Db, BR, SR, Nb, w):
    bx = (RPM**(1.02 - 0.02*Nb)) / (RPM**0.92)
    CCS = UCS
    
    
    
    #ROP = (w[0]*WOB**w[1]*RPM**w[2]*math.cos(math.radians(SR))*bx) / (CCS**w[3]*Db*math.tan(math.radians(BR)))
    ROP = ((w[0]*(WOB**w[1])*(RPM**w[2])*math.cos(math.radians(SR))) / ((CCS**w[3])*Db*math.tan(math.radians(BR))))*(60/12)
    return ROP


'''    BR = 15 + 41
    SR = 1
    WOB = WOB / 10
    CCS = 28000'''

def Objective_Function(w):
    
    Error = 0
    ROP_pred_list = []
    ROP_pred_list = [Kerkar_Model(UCS, WOB, RPM, Db, BR, SR, Nb, w) for UCS, WOB, RPM, Db, BR, SR, Nb in zip(UCS, WOB, RPM, Db, BR, SR, Nb)]
    Error = [((abs(ROP_Data - ROP_pred))/ROP_Data) for ROP_Data, ROP_pred in zip(ROP_Data, ROP_pred_list)] 
    Ave_Error = sum(Error) / len(ROP_Data)
    return Ave_Error

def De_Algorithm(fobj, bounds, mut=0.8, crossp=0.7, popsize=100, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
                    
                    
        #print("Iteration number= %s" % (i))
        #print("Best Fitness= %s" % (fitness[best_idx]))
        #print("Best values= %s" % (best))
        yield best, fitness[best_idx]

def Run_DEA(ite):
    results = []
    result = list(De_Algorithm(Objective_Function, 
                 [(0.001, 0.1),    
                  (0, 2),
                  (0, 2),
                  (0, 2)], 
                  mut=0.7, crossp=0.8, popsize=15, its=ite))
    
    df = pd.DataFrame(result)
    return results, df


def Best_coffs(df):
    
    df['w1'], df['w2'], df['w3'], df['w4']  = zip(*df[0]) # Unzip
    cols = [0] # Drop the first column
    df.drop(df.columns[cols],axis = 1,inplace = True) # Drop the first column
    df.columns.values[0] = "Fitness" # name the first column as Fitness
    best_coff = df.iloc[len(df)-1,1:] # insert the best coefficients into the best_coff
    
    return best_coff

def Plot_DEA_Evolution(df):
    
    data_ncol=len(df.columns) # number of paramters 
    fig = plt.figure(figsize=(15,15)) # you may change these to change the distance between plots.

    for i in range(1,(data_ncol+1)):
        if i<data_ncol:
            plt.subplot(3, 3, i)
            plt.plot(df['w{}'.format(i)],'bo', markersize=4)
            plt.xlabel('Iteration')
            plt.ylabel('w{}'.format(i))
            plt.grid(True)
        else:       
            plt.subplot(3, 3, data_ncol)
            plt.plot(df['Fitness'],'red', markersize=4)
            plt.xlabel('Iteration')
            plt.ylabel('Fitness')
            plt.grid(True)
    plt.show()

def Plot_variables(x, y, xlab, ylab, xmin, xmax, ymin, ymax):
    
    fig = plt.figure(figsize=(7,7))
    plt.scatter(x, y)
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.xlabel(xlab)
    plt.ylabel(ylab)


# ### Data visualization

# In[ ]:


#Plot_data_seaborn()


# ### Run program

# In[ ]:


UCS, WOB, RPM, Db, BR, SR, Nb, ROP_Data = Turn_data_to_seperate_lists()
results, df = Run_DEA(300)
best_coff = Best_coffs(df)


# In[ ]:


print(best_coff)


# ### DEA agents visualization

# In[ ]:


Plot_DEA_Evolution(df)


# In[ ]:


Est_ROP_Model = [Kerkar_Model(UCS, WOB, RPM, Db, BR, SR, Nb, best_coff) for UCS, WOB, RPM, Db, BR, SR, Nb in zip(UCS, WOB, RPM, Db, BR, SR, Nb)]
Est_ROP_Model = pd.DataFrame(Est_ROP_Model)
ROP_Data = pd.DataFrame(ROP_Data)


# In[ ]:


Plot_variables(ROP_Data, Est_ROP_Model, 'ROP Data ft/hr', 'ROP Model ft/hr', 0, 60, 0, 60)


# In[ ]:


fig = plt.figure(figsize=(7,7))
plt.scatter(WOB[0:9], ROP_Data[0:9])
plt.plot(WOB[0:9], Est_ROP_Model[0:9])

plt.scatter(WOB[9:18], ROP_Data[9:18])
plt.plot(WOB[9:18], Est_ROP_Model[9:18])

plt.scatter(WOB[19:27], ROP_Data[19:27])
plt.plot(WOB[19:27], Est_ROP_Model[19:27])

plt.scatter(WOB[27:33], ROP_Data[27:33])
plt.plot(WOB[27:33], Est_ROP_Model[27:33])

plt.scatter(WOB[33:39], ROP_Data[33:39])
plt.plot(WOB[33:39], Est_ROP_Model[33:39])

plt.xlabel("WOB lbf")
plt.ylabel("ROP ft/hr")


# In[ ]:


Est_ROP_Model.to_csv('Est_ROP_Model_csv_to_submit.csv', index = False)


# In[ ]:


def Kerkar_Model(UCS, WOB, RPM, Db, BR, SR, Nb, w):
    bx = (RPM**(1.02 - 0.02*Nb)) / (RPM**0.92)
    CCS = UCS

    
    #ROP = (w[0]*WOB**w[1]*RPM**w[2]*math.cos(math.radians(SR))*bx) / (CCS**w[3]*Db*math.tan(math.radians(BR)))
    ROP = ((w[0]*(WOB**w[1])*(RPM**w[2])*math.cos(math.radians(SR))) / ((CCS**w[3])*Db*math.tan(math.radians(BR))))*(60/12)
    return ROP


UCS = 28000
WOB = 230
RPM = 100
Db = 3.75
BR = 15 + 41
SR = 1
Nb = 4
w = [0.005804, 2.61, 0.7, 0.97]

Kerkar_Model(UCS, WOB, RPM, Db, BR, SR, Nb, w)


# In[ ]:


import pandas as pd
SNL 2015 SWG data2 = pd.read_csv("../input/SNL 2015 SWG data2.csv")

