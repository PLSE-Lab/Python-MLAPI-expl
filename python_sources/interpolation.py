#!/usr/bin/env python
# coding: utf-8
xov=min(k1)
for i in np.arange(len(x)):
    if k1[i]==xov:
        xo= x[i]x:    10       20     30     40      50     60
cosx: .9849   .9397  .866   .7660   .6428  .5000
    
# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from pandas import Series,DataFrame
sns.set_style("whitegrid")
import math
import scipy


# In[ ]:


X=25    #i will be entering and altering this value as per my demands


# In[ ]:


x=[10,20,30,40,50,60]
y=[.9849,.9397,.866,.7660,.6428,.5000]


# In[ ]:


h=x[1]-x[0]
yo=y[0]
xo=x[0]
r=(X-xo)/h
k1=[]
for i in np.arange(len(x)):
    k1=np.append(k1,abs(x[i]-X))

def diff(d):
    k2=[]
    for i in np.arange(len(d)-1):
            k2=np.append(k2,d[i+1]-d[i])
        
    return k2

y1=diff(y)
y2=diff(y1)
y3=diff(y2)
y4=diff(y3)
y5=diff(y4)

Y=yo+r*y1[0]+r*(r-1)*y2[0]/math.factorial(2)+r*(r-1)*(r-2)*y3[0]/math.factorial(3)+r*(r-1)*(r-2)*(r-3)*y4[0]/math.factorial(4)+r*(r-1)*(r-2)*(r-3)*(r-4)*y5[0]/math.factorial(5)

print(Y)


# In[ ]:





# In[ ]:


import numpy as np


# In[ ]:


np.random.seed(2)


# In[ ]:


stats = {}
line = input()
while line:
     (wsets,lsets,wgames,lgames) = (0,0,0,0)
     (winner,loser,setscores) = line.strip().split(':',2)
     sets = setscores.split(',')
     for set in sets:
         (winstr,losestr) = set.split('-')
         win = int(winstr)
         lose = int(losestr)
         wgames = wgames + win
         lgames = lgames + lose
         if win > lose:
             wsets = wsets + 1
         else:
            lsets = lsets + 1
        
    for player in [winner,loser]:
        try: 
            stats[player]
         except keyError:
             stats[player] = [0,0,0,0,0,0]
     if wsets >= 3:
         stats[winner][0] = stats[winner][0] + 1
     else:
         stats[winner][1] = stats[winner][1] + 1
     stats[winner][2] = stats[winner][2] + wsets 
     stats[winner][3] = stats[winner][3] + wgames
     stats[winner][4] = stats[winner][4] - lsets
     stats[winner][5] = stats[winner][5] - lgames
     stats[loser][2] = stats[loser][2] + lsets
     stats[loser][3] = stats[loser][3] + lgames
     stats[loser][4] = stats[loser][4] - wsets
     stats[loser][5] = stats[loser][5] - wgames
     line = input()
statlist = [(stat[0],stat[1],stat[2],stat[3],stat[4],stat[5],name) for name in
stats.keys() for stat in [stats[name]]]

statlist.sort(reverse = True)

for entry in statlist:
    print(entry[6],entry[0],entry[1],entry[2],entry[3],-entry[4],-entry[5])


# In[ ]:


stats = {}
line = input()


# In[ ]:


while line:
     (wsets,lsets,wgames,lgames) = (0,0,0,0)
     (winner,loser,setscores) = line.strip().split(':',2)
     sets = setscores.split(',')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




