#!/usr/bin/env python
# coding: utf-8

# Covariances aren't probabilities. Let's try if we can generate usefull features if we treat them as probs anyway.
# A covariance matrix is build from the fnc data. The covariances are scaled to a range between 0 and 1 and to sum up to 1 per Id. Then the convertet covariances are treated as transition probabilities and the steady state are being calculated and saved as features per Id.

# In[ ]:


import pandas as pd
import numpy as np
import re
import tqdm
from pathlib import Path


# In[ ]:


kaggle_input_path = Path('/kaggle/input/trends-assessment-prediction') # '/kaggle/input/trends-assessment-prediction'


# In[ ]:


icn = pd.read_csv(kaggle_input_path/'ICN_numbers.csv').values.flatten()

f_data = pd.read_csv(kaggle_input_path/'fnc.csv')
f_data = f_data.set_index('Id')
f_data = f_data.T.reset_index()

f_data.head()


# In[ ]:


# regex from https://www.kaggle.com/kpriyanshu256/trends-graph?scriptVersionId=36333263

f_data['x'] = f_data['index'].apply(lambda x: int(re.findall(r'(?<=\().*?(?=\))', x)[0]))
f_data['y'] = f_data['index'].apply(lambda x: int(re.findall(r'(?<=\().*?(?=\))', x)[1]))
f_data.drop('index', inplace=True,axis=1)


# In[ ]:


def scale(trans_mx):
    trans_mx = (trans_mx/ trans_mx.min(axis=1))/(trans_mx.max(axis=1)-trans_mx.min(axis=1))
    trans_mx = trans_mx / trans_mx.sum(axis=1)
    return trans_mx

# https://stackoverflow.com/questions/52137856/steady-state-probabilities-markov-chain-python-implementation
def steady_state_prop(p):
    dim = p.shape[0]
    q = (p-np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q,ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)

    return np.linalg.solve(QTQ,bQT)


# In[ ]:


fnc_steady_state = None

ids = list(set(f_data.columns.values)-set(['x','y']) )

for i in tqdm.tqdm(ids, total=len(ids)):
    
    cov_mx = np.zeros((53,53))

    for j, r in f_data[[i, 'x', 'y']].iterrows():
        x=np.argwhere(icn==int(r['x']))[0][0]-1
        y=np.argwhere(icn==int(r['y']))[0][0]-1
        cov_mx[x,y] = r[i] 
        cov_mx[y,x] = r[i] 

    cov_mx_scaled = scale(cov_mx)
    
    ssp = steady_state_prop(cov_mx_scaled).reshape(1,53)
    
    if not isinstance(fnc_steady_state, pd.DataFrame):
        fnc_steady_state = pd.DataFrame(ssp, columns=icn)
    else:
        fnc_steady_state = fnc_steady_state.append(pd.DataFrame(ssp, columns=icn))
        

fnc_steady_state['Id'] = ids


# In[ ]:


display(fnc_steady_state.head())


# In[ ]:


fnc_steady_state.to_csv('fnc_steady_state.csv', index = False)


# In[ ]:




