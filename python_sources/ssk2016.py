#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sklearn
import catboost
import lightgbm
import xgboost
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


Data = pd.read_csv('../input/test_data.csv').values


# In[ ]:


def ml(a, b):
    if a[0] < b[0]:
        if a[2] < b[0]:
            x = 0
        elif a[2] < b[2]:
            x = a[2] - b[0]
        else:
            x = b[2] - b[0]
    elif a[0] < b[2]:
        if a[2] < b[2]:
            x = a[2] - a[0]
        else:
            x = b[2] - a[0]
    else:
        x = 0
    if a[1] < b[1]:
        if a[3] < b[1]:
            y = 0
        elif a[3] < b[3]:
            y = a[3] - b[1]
        else:
            y = b[3] - b[1]
    elif a[1] < b[3]:
        if a[3] < b[3]:
            y = a[3] - a[1]
        else:
            y = b[3] - a[1]
    else:
        y = 0
    S_a = (a[2] - a[0] + 1)*(a[3] - a[1] + 1)
    S_b = (b[2] - b[0] + 1)*(b[3] - b[1] + 1)
    S_ab = x*y
    return S_ab/(S_a + S_b - S_ab)


# In[ ]:


Iz = np.sort(list(set(Data[:,1])))


# In[ ]:


Answer = np.zeros([len(Iz),5], dtype= 'int32')
for i in range(len(Iz)):
    B = Data[Data[:,1] == Iz[i]][:,2:]
    S = 0
    Y = 0
    d = 10
    for x_min in np.arange(np.min(B[:,0]),np.max(B[:,2]),d):
        for y_min in np.arange(np.min(B[:,1]),np.max(B[:,3]),d):
            for x_max in np.arange(x_min,np.max(B[:,2]),d):
                for y_max in np.arange(y_min,np.max(B[:,3]),d):
                    X = [x_min, y_min, x_max, y_max]
                    s = 0
                    for j in range(len(B)):
                        s = s + ml(X,B[j])
                    if s > S:
                        Y = [x_min, y_min, x_max, y_max]
                        S = 0 + s
    print(i, S/len(B))
    Answer[i,0] = Iz[i]
    Answer[i,1] = Y[0]
    Answer[i,2] = Y[1]
    Answer[i,3] = Y[2]
    Answer[i,4] = Y[3]


# In[ ]:


pd.DataFrame(Answer).to_csv('Answer.csv', header = False, index = False)


# In[ ]:





# In[ ]:




