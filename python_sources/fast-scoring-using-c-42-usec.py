#!/usr/bin/env python
# coding: utf-8

# Let's use a C implementation of the score function to speed up evaluation

# In[ ]:


get_ipython().run_cell_magic('writefile', 'score.c', '\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n\n#define NF 5000\nint cost[NF][101];\nint fs[NF];\n\nint cf[NF][10];\n\nint loaded=0;\n\nfloat acc[301][301];\n\nvoid precompute_acc() {\n    \nfor(int i=125;i<=300;i++) \n    for(int j=125;j<=300;j++)\n      acc[i][j] = (i-125.0)/400.0 * pow(i , 0.5 + fabs(i-j) / 50 );    \n}\n\nvoid read_fam() {\n  FILE *f;\n  char s[1000];\n  int d[101],fid,n;\n  int *c;\n\n  f=fopen("../input/santa-workshop-tour-2019/family_data.csv","r");\n  if (fgets(s,1000,f)==NULL)\n    exit(-1);\n\n  for(int i=0;i<5000;i++) {\n    c = &cf[i][0];\n    if (fscanf(f,"%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",\n               &fid,&c[0],&c[1],&c[2],&c[3],&c[4],&c[5],&c[6],&c[7],&c[8],&c[9],&fs[i])!=12)\n      exit(-1);\n\n    //    printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\\n",\n    //fid,c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],fs[i]);\n    n = fs[i];\n\n    for(int j=1;j<=100;j++) {\n      if (j==c[0]) cost[i][j]=0;\n      else if (j==c[1]) cost[i][j]=50;\n      else if (j==c[2]) cost[i][j]=50 + 9 * n;\n      else if (j==c[3]) cost[i][j]=100 + 9 * n;\n      else if (j==c[4]) cost[i][j]=200 + 9 * n;\n      else if (j==c[5]) cost[i][j]=200 + 18 * n;\n      else if (j==c[6]) cost[i][j]=300 + 18 * n;\n      else if (j==c[7]) cost[i][j]=300 + 36 * n;\n      else if (j==c[8]) cost[i][j]=400 + 36 * n;\n      else if (j==c[9]) cost[i][j]=500 + 36 * n + 199 * n;\n      else cost[i][j]=500 + 36 * n + 398 * n;\n    }\n  }\n\n}\n\nfloat max_cost=1000000000;\n\nint day_occ[102];\n\nstatic inline int day_occ_ok(int d) {\n  return !(d <125 || d>300);\n}\n\nfloat score(int *pred) {\n  float r=0;\n    \n  if (!loaded) {\n      read_fam();\n      precompute_acc();\n      loaded = 1;\n  }\n\n  // validate day occupancy\n  memset(day_occ,0,101*sizeof(int));\n\n  for(int i=0;i<NF;i++) {\n    day_occ[pred[i]]+=fs[i];\n    r+=cost[i][pred[i]];\n  }\n       \n  day_occ[101]=day_occ[100];\n\n  for (int d=1;d<=100;d++) {\n    if (day_occ[d]<125 || day_occ[d]>300) return max_cost;\n    r += acc[day_occ[d]][day_occ[d+1]];\n  }\n  return r;\n}  ')


# In[ ]:


get_ipython().system('gcc -O5 -shared -Wl,-soname,score     -o score.so     -fPIC score.c')
get_ipython().system('ls -l score.so')


# In[ ]:


# Let's import the score function in python
import ctypes
from numpy.ctypeslib import ndpointer

lib = ctypes.CDLL('./score.so')
score = lib.score
# Define the types of the output and arguments of this function.
score.restype = ctypes.c_float
score.argtypes = [ndpointer(ctypes.c_int)]


# In[ ]:


# From now on, below is pure basic python with greedy approach

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

sub = pd.read_csv('../input/santa-workshop-tour-2019/sample_submission.csv')
pred = np.int32(sub.assigned_day.values)
score(pred)


# In[ ]:


# fast enough ? ;-) 
get_ipython().run_line_magic('timeit', 'score(pred)')


# In[ ]:


fam = pd.read_csv("../input/santa-workshop-tour-2019/family_data.csv")
# preferred day per familly
pref = fam.values[:,1:-1]
n_people = fam.n_people.values
fam_size_order = np.argsort(n_people)#[::-1]


# In[ ]:


# First try to assign preferred days to each family, repeat 20 times

best_score = score(pred)

for t in tqdm(range(20)):
    print(t,best_score,'     ',end='\r')
    for i in fam_size_order:
        for j in range(10):
            di = pred[i]
            pred[i] = pref[i,j]
            cur_score = score(pred)
            if cur_score < best_score:
                best_score = cur_score
            else:
                pred[i] = di


# In[ ]:


# Then try to trade days between families by pair

def opt():
    best_score = score(pred)
    for i in tqdm(range(5000)):
        if (i%10==0):
            print(i,best_score,'     ',end='\r')
        for j in range(5000):
            di = pred[i]
            pred[i] = pred[j]
            pred[j] = di
            cur_score = score(pred)
            if cur_score < best_score:
                best_score = cur_score
            else: # revert
                pred[j] = pred[i]
                pred[i] = di
            
opt()
opt()
            


# In[ ]:


sub.assigned_day = pred
_score = score(pred)

sub.to_csv(f'submission_{_score}.csv',index=False)


# In[ ]:


get_ipython().system('ls -l')


# In[ ]:




